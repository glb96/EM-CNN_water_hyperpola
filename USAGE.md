# Installation and usage

This guide describes the complete EM-CNN example: XYZ configurations → molecular pickle data → electrostatic maps → normalized datasets → CNN training → evaluation. The associated manuscript is introduced in [README.md](README.md).

The two scripts in `Example/` are ordinary Python execution examples, independent of a cluster scheduler. Their configuration blocks and named function arguments show which parameters are used at each step.

## 1. Install the environment

Use Python 3.11 or newer and an isolated environment. For example, on Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch using the command appropriate to your operating system and GPU from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then, from the repository root:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU available:', torch.cuda.is_available())"
```

Training uses [Keras 3 with the PyTorch backend](https://keras.io/getting_started/). The scripts select this backend before importing Keras. TensorFlow and fidle are not required by this version of the workflow. On a cluster, activate your environment within your own GPU allocation; no scheduler directives are assumed here.

The requirements specify broad compatibility bounds rather than reproducing a frozen historical environment. A small end-to-end CPU execution check was performed with Python 3.12, NumPy 2.3.5, PyTorch 2.15.0+cpu, Keras 3.15.1, and TensorBoard 2.21.0. This verifies execution, not full training convergence or GPU performance.

## 2. Run the supplied example

From the repository root:

```bash
# CPU preparation: conversion, maps, augmentation, normalization, rotation metadata.
python Example/prepare_data.py

# GPU execution: training followed by evaluation of the best checkpoint.
python Example/run_cnn.py
```

Paths are resolved relative to the scripts, so they do not depend on your working directory. Default outputs are `Example/prepared/` and `Example/cnn_run/`. The scripts require an empty output directory for a new preparation or training run, preventing accidental mixing of experiments. Select different locations with:

```bash
python Example/prepare_data.py --input-dir Example --output-dir Example/prepared_trial --workers 4
python Example/run_cnn.py --prepared-dir Example/prepared_trial --run-dir Example/cnn_trial
```

Move the complete prepared directory and the code to the GPU machine if preparation and training run on different systems. The recorded source path in metadata is informational; evaluation does not require the original machine's absolute paths.

The example often referred to as `1000-200` contains the following **actual file counts**:

| Input | Original configurations | Prepared samples with default x/y augmentation | Role |
| --- | ---: | ---: | --- |
| `Example/train.xyz` | 800 | 3200 | Gradient-based learning. |
| `Example/test.xyz` | 200 | 800 | Validation loss during training, checkpoint selection and learning-rate adjustment. |
| `Example/valid.xyz` | 200 | 800 | Separate final assessment after training. |

Thus the training and training-monitoring files together contain 1000 configurations, with another 200 reserved for final assessment. These filenames are historical: `test` serves as the optimization validation set, whereas `valid` is the held-out evaluation set.

The `.p` and rotation files already in `Example/` illustrate the intermediate formats. The execution example starts from the three XYZ files and regenerates everything under the chosen output directory; it does not overwrite or rely on the bundled pickle files.

## 3. Understand the XYZ input

This is the essential input contract for using your own configurations. Each frame describes **one target water molecule and its explicit environment**, with its reference tensor. A simulation snapshot is not automatically expanded into one sample per water molecule.

A multi-frame XYZ consists of successive blocks:

1. Total number of atoms in the frame.
2. One header line containing `l3_tensor="_JSON ..."`.
3. Exactly that many atom lines, with `element x y z` as the first four whitespace-separated fields.

| Atom lines | Contents |
| --- | --- |
| 1–3 | Target molecule in **O H H** order. |
| 4–6 | First surrounding water, in **O H H** order. |
| 7–9, etc. | Further surrounding waters, each in **O H H** order. |

The first target hydrogen is H1 and the second is H2. Their order defines the molecular frame and must remain consistent. Every environment triplet is checked. The reader expects at least six atoms, with `(number of atoms - 3)` divisible by three. Additional atom columns and unrelated header fields are ignored.

Coordinates are in **angstroms in the laboratory frame**. Supply whole water molecules and the appropriate periodic images around the target. The reader does not reconstruct periodic boundaries, choose a neighbor cutoff, or read charges from XYZ columns. A different environment definition can change the field and the predictions.

### Tensor header

`l3_tensor` is the target molecule's **laboratory-frame** first hyperpolarizability, using the same axes as the coordinates. It is a finite JSON array with shape `(3, 3, 3)` and indexing `beta[i][j][k]`, where each index follows `x, y, z`. All 27 entries must be present. The model retains 18 components using the interchange symmetry of the last two indices; it does not impose full Kleinman symmetry.

Use atomic units for beta when using the supplied normalization constants. No beta unit conversion is performed. The current supervised workflow requires reference beta values for dataset preparation and evaluation; coordinate-only inference is not provided by these two examples.

Minimal syntax illustration (the zero tensor is deliberately artificial, not a training reference):

```text
6
l3_tensor="_JSON [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]"
O  0.000000  0.000000  0.000000
H  0.757000  0.000000  0.586000
H -0.757000  0.000000  0.586000
O  3.000000  0.000000  0.000000
H  3.757000  0.000000  0.586000
H  2.243000  0.000000  0.586000
```

Keep the complete tensor on the single header line. The real files in `Example/` demonstrate this format with physical reference tensors. Additional data and the final trained Keras model will be provided very soon on Zenodo; the record link will be added once available.

## 4. Convert XYZ to molecular pickle data

**Implementation:** `Scripts/xyz_to_pickle_with_rotations.py`.

**Input:** one XYZ file per split. **Output:** a configuration `.p` file, and optionally a separate file of transformation matrices and reference metadata.

The first stage of `Example/prepare_data.py` calls:

```python
xyz_to_pickle(
    input_xyz=str(input_dir / f'{split}.xyz'),
    output_p=str(raw_path),
    rotation_output=str(raw_rotations) if raw_rotations else None,
    water_charges=water_charges,
)
```

### Charges are chosen here

`WATER_CHARGES = (-0.820, 0.410, 0.410)` in `Example/prepare_data.py` sets the O/H1/H2 environment charges in elementary-charge units. They are written to `env_charges` in the pickle file. The uniform extra field is zero in this converter. The target water's own charges are not part of the environment sum.

**If you change the water charge model, change these charges before conversion and regenerate the maps and prepared data.** The grid generator knows the numerical charges through `env_charges`; it does not infer a water model or assign charges itself. The present converter supports three-site water with O/H/H triplets. Four-site models, ions, other molecules, or nonzero external fields require adapting the conversion and input representation, not just changing three numbers.

### Molecular coordinates and separate transformations

The target oxygen becomes the origin. The frame is constructed using:

- `ex`: normalized H1-to-H2 vector.
- `ez0`: normalized O-to-hydrogen-midpoint vector.
- `ey`: normalized `ex × ez0`.
- `ez = ey × ex`.

With these basis vectors as rows of `R`, column vectors transform as `v_mol = R @ v_lab`, after translating positions by the oxygen coordinate. The tensor transforms with a factor of `R` on each of its three indices.

This is intentionally the **existing left-handed convention**, `det(R) = -1`. Its inverse is `R.T`. Do not change signs or cross-product order when reusing existing model weights or datasets.

EM-CNN learns entirely in the molecular frame. The molecular configuration pickle therefore contains everything needed to calculate maps and train the network; the frame transformations are kept separately because they are only needed later to recover and evaluate laboratory-frame responses. The example saves them for `valid` only, without adding them as CNN input features.

### Pickle contents

The configuration dictionary contains `T_steps`, `N_molecule`, and entries indexed by `(t, n)`. The XYZ converter uses one target per frame: `N_molecule = 1` and keys `(t, 0)`.

| Per-configuration key | Shape / value |
| --- | --- |
| `atoms` | `['O', 'H1', 'H2']` |
| `pos_mol_interest` | `(3, 3)` target coordinates, molecular frame, angstroms. |
| `env_positions` | `(N_environment_waters, 3, 3)` environment coordinates, same frame and units. |
| `env_charges` | `(N_environment_waters, 3)` corresponding charges. |
| `field_value` | `(3,)` extra molecular-frame field in atomic units; zero here. |
| `beta` | `(3, 3, 3)` reference molecular tensor. |

Raw rotation metadata stores `R_lab_to_mol`, the 18 molecular reference components, and the source indices. It is tagged `format='cnn_validation_rotations_v1'`, `stage='raw'`.

## 5. Build the electrostatic maps

**Implementation:** `Scripts/create_electrostatic_grid.py`, function `generate_maps`.

**Input:** molecular configuration pickle. **Output:** grid pickle containing the fields, tensor targets, grid coordinates, and settings. Visualization XYZ files are written alongside it.

For each grid point, the field sums the regularized contributions of the supplied environment charges:

$$
\mathbf E(\mathbf r)=a_0^2\sum_a q_a\frac{\mathbf r-\mathbf r_a}{\bigl(|\mathbf r-\mathbf r_a|^2+s^2\bigr)^{3/2}}+\mathbf E_{\mathrm{uniform}}.
$$

Distances and `s` are in angstroms; `a0 = 0.529177` is the bohr-to-angstrom conversion factor. The output field is in atomic units. A mask retains the region around the target molecule; values outside the mask remain zero.

### Spatial extent, smoothing, and mesh resolution are different settings

All of these are visible in the configuration block of `Example/prepare_data.py`:

| Setting | Default | Effect |
| --- | ---: | --- |
| `SMOOTHING` | 0.6 Å | Regularization length `s`, controlling the field close to point charges. |
| `MASK_EXTENSION` | 2.4 Å | Extent of the region sampled around the target atoms. |
| `BOX_HALF_WIDTH` | 6.0 Å | Initial coordinate box is `[-6, 6]` along each axis. It must contain the mask with zero-valued boundaries. |
| `GRID_POINTS` | 35 | Points per axis before cropping; mesh spacing is `2 * BOX_HALF_WIDTH / (GRID_POINTS - 1)`. |
| `WORKERS` | 1 | Number of CPU processes; adapt to your allocation and RAM. |

The mask is the union of spheres around H1, H2, O, and the arithmetic mean of the three target positions. Their radii are `0.8 + MASK_EXTENSION`, `0.8 + MASK_EXTENSION`, `1.0 + MASK_EXTENSION`, and `1.1 + MASK_EXTENSION`, respectively. The initial grid is then cropped around the mask with boundary padding. For the supplied example and defaults, the final dimensions are **25 × 21 × 23**, with three field channels.

The execution example builds the mask from the **first training configuration**, and reuses this reference for all splits. It is a fixed molecular grid, not a separately fitted mask around every sample. Configurations must remain compatible with that molecular geometry and convention.

For exploring these hyperparameters, prioritize **smoothing** and **spatial extent of the sampled region**. In the author's tests on this water system, mesh convergence was relatively quick, while refinement became expensive. Increasing the point count raises field storage and convolution cost rapidly; doubling it along all axes roughly multiplies the initial voxel count by eight. Changing the retained region can also change the cropped shape and dense-layer parameter count. Check mesh convergence for a new system, but avoid treating an extremely fine mesh as the default route to improved predictions.

Example of a separate preparation experiment:

```bash
python Example/prepare_data.py --smoothing 0.8 --mask-extension 2.4 --grid-points 35 --output-dir Example/prepared_s08
```

Any change in charge model, smoothing, grid coordinates, mask, or normalization changes the input representation. Retrain or use a model explicitly trained for that representation; matching array dimensions alone is insufficient.

`molecule_interest.xyz`, `grid_original.xyz`, and `grid_masked.xyz` are **visualization outputs**, not new input datasets. The extra `X` point in the target visualization marks the mean target position, and the grid XYZ files use dummy element labels for points.

## 6. Prepare the CNN datasets

**Implementation:** `Scripts/prepare_data_set_CNN_new.py`, function `prepare_data_for_GNN_ready_to_eat` (a historical name; it prepares CNN data).

**Input:** one grid pickle per split. **Output:** `train_CNN.p`, `test_CNN.p`, and `valid_CNN.p`, plus normalization files.

The preparation performs these operations:

1. Read the maps, optionally using disk-backed arrays for processing.
2. Optionally keep the first fraction of each split (`RATIO_DATA_TO_USE`, default `1.0`). This is not random subsampling.
3. Leave the field representation unchanged by default (`USE_R2_1=False`, `USE_LOG_MEAN=False`).
4. Apply x/y symmetry augmentation to both the spatial/vector field representation and the rank-three tensor.
5. Normalize the fields and tensor components.
6. Transpose fields to channels-last order and save a PyTorch `TensorDataset`.

With both symmetries enabled, each configuration produces four consecutive rows: **identity, x, y, xy**. No random shuffling occurs here. The input splits already exist before augmentation, and related augmented rows stay in the same split.

The field means and standard deviations are determined from the augmented training fields only, then reused for `test` and `valid`. The original masking behavior is retained: field-statistic estimation ignores near-zero values and field normalization leaves exact zeros unchanged. Beta values use the supplied fixed `beta_mean_ref` and `beta_std_ref` constants. These constants are specific to the reference dataset; new physical datasets may need their own training-derived beta statistics.

The example saves these exact beta constants as `beta_reference.py`, and the complete statistics as `*_CNN_normalization_stats.p`. Evaluation reads the saved constants from this preparation, rather than relying on a missing external `eval.py`. A future pretrained model must likewise be used with its original field and beta normalization statistics.

Prepared inputs have shape `(N, Nx, Ny, Nz, 3)` and targets `(N, 18)`. The target order is:

```text
xxx xxy xxz xyy xyz xzz
 yxx yxy yxz yyy yyz yzz
 zxx zxy zxz zyy zyz zzz
```

Temporary memmap filenames are shared within each working directory. The execution example processes splits sequentially in separate subdirectories. Disk-backed processing reduces some allocations but does not make the whole pipeline memory-free: raw map loading and final tensor creation still require RAM.

### Align the validation transformations

`Scripts/augment_validation_rotations.py` applies the same symmetries and sample selection to the separate validation metadata, producing `valid_rot_aug.p`. Its row order therefore matches `valid_CNN.p` exactly. Even if augmentation is disabled, the example converts the raw metadata to the evaluation format using `augmentation='none'`.

The final `preparation.json` records input counts, prepared counts, grid settings, augmentation, and normalization. Its presence indicates that the preparation script reached the end successfully.

## 7. Train the CNN on GPU

**Implementation:** `Scripts/train_CNN_LR_batch.py`, function `train_model`; called by `Example/run_cnn.py`.

**Input:** `train_CNN.p` and `test_CNN.p`. **Output:** model checkpoints, a final model, and TensorBoard logs.

The architecture and default training hyperparameters are those selected by the author as performing well in the explored benchmark. They are provided as a starting point, not a guarantee of optimality for every dataset.

The network has three 3D convolutional layers (128, 128, and 64 filters), each preceded by average pooling, followed by dense layers with 256, 512, 1024, 1024, and 512 units. Hidden layers use ReLU; the 18 outputs are linear. Training minimizes MSE on normalized beta values using Adam with gradient clipping.

| Phase | Maximum epochs | Batch size | Initial learning rate |
| --- | ---: | ---: | ---: |
| 1 | 20 | 256 | 0.001 |
| 2 | 15 | 64 | 0.0005 |
| 3 | 30 | 128 | 0.00025 |
| 4 | 30 | 64 | 0.0001 |
| 5 | 75 | 128 | 0.0001 |

Each phase starts a separate fitting call with its specified learning rate. Learning-rate reduction and early stopping monitor `test` loss within the phase; an early stop ends that phase, after which the next phase begins. The best-model checkpoint tracks the best monitored loss across the run. Epoch file names and TensorBoard steps are numbered cumulatively across phases.

**No training shuffling is enabled by default**, matching the original script: `SHUFFLE_TRAIN=False`. If you explicitly set it to `True`, the training DataLoader shuffles samples; test and final evaluation ordering remain unchanged. `SEED=42` seeds model initialization and supported random operations; exact numerical reproducibility can still depend on hardware and software versions.

This step is intended for a GPU. `run_cnn.py` checks for a CUDA device and otherwise stops, unless `--allow-cpu` is supplied for a small execution check. CPU preparation also needs PyTorch to save datasets, but does not require a GPU.

The run directory contains:

| Path | Contents |
| --- | --- |
| `models/best-model.keras` | Checkpoint selected by the lowest monitored loss. |
| `models/001.keras`, etc. | Per-epoch checkpoints. |
| `final_model.keras` | Model state after the last training phase; not necessarily the best checkpoint. |
| `logs/` | TensorBoard training/validation curves. |
| `training.json` | Schedule, seed, shuffle setting, and execution information. |
| `preparation.json` | Copy of the preparation settings used for this run. |

View the curves with `tensorboard --logdir Example/cnn_run/logs`.

## 8. Evaluate the trained model

**Implementation:** `Scripts/eval_labo.py`; automatically called by `Example/run_cnn.py` after training.

**Inputs:** the best checkpoint, `valid_CNN.p`, aligned validation transformations, and the saved beta constants. **Outputs:** reports and prediction arrays under `cnn_run/evaluation/`.

Evaluation loads `models/best-model.keras`, predicts molecular tensors, reverses normalization, and transforms both predictions and references to the laboratory frame. The validation dataset is not used to select the training checkpoint. Reference labels and rotation metadata are checked for consistency before prediction.

Reports include global and component-wise R2, MSE, and MAE, in molecular and laboratory frames, using either the 18 independent entries or all 27 tensor entries. The global R2 uses component-centered reference variances and is not the arithmetic mean of component R2 values. The custom training-time R2 is batch-centered on normalized values, so use the separate physical-unit evaluation for reporting performance.

Laboratory reports also give the component average of `RMSE / sigma_ref`, with reference standard deviations computed using `ddof=0`. For an individual component with nonzero variance, this ratio is `sqrt(1 - R2_component)`.

A baseline predicts the same saved mean molecular tensor for every sample. Its laboratory-frame tensor changes with molecular orientation. This distinguishes the contribution of orientation from the learned environment-dependent molecular variations; its molecular R2 need not be exactly zero on held-out data.

| Output suffix (prefix `validation_labo`) | Contents |
| --- | --- |
| `.json` | Settings, model metrics, and constant-tensor baseline metrics. |
| `_components.csv` | Model component metrics. |
| `_baselines_components.csv` | Baseline component metrics. |
| `_lab_normalized_rmse.csv` | Component RMSE, reference standard deviation, and their ratios. |
| `_predictions.npz` | Molecular/laboratory references and predictions, identifiers, and baseline arrays. |

`EVALUATION_SAMPLES='all'` scores the augmented validation set. Set it to `'original'` in `run_cnn.py` to report only the original rows. To repeat evaluation without retraining:

```bash
python Example/run_cnn.py --evaluate-only
```

For a brief CPU execution check on a small input subset, prepare that subset with `--input-dir`, then run `run_cnn.py --allow-cpu --smoke-test` with the corresponding `--prepared-dir` and a fresh `--run-dir`. This performs one epoch and evaluation; it is not a benchmark or a substitute for the full schedule.

## 9. Files to keep and cleanup

All **six Python modules in `Scripts/`** have a role in this workflow. `augment_validation_rotations.py` transforms metadata while `eval_labo.py` evaluates tensors; these are complementary, not duplicates. The old slow `built_electrostatic_potential` function inside the grid module is not called by this workflow; the fast field implementation is used instead.

The legacy `submit_*.sh` files are not used and may be removed from the repository. They are site-specific; in the old version `submit_prepare_data.sh` repeats grid generation and resubmits itself, and `submit_eval_labo.sh` refers to a different filename. Use the two execution examples instead.

For further experiments, keep the prepared datasets together with their normalization, rotation metadata, and settings. Keep trained model weights with those settings as well. Raw `.p`, map `.p`, and visualization files under the per-split preparation directories are useful for inspection but are not read by `run_cnn.py` once preparation is complete. The bundled example pickle files can also be regenerated from the XYZ files.
