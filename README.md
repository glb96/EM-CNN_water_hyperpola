# EM-CNN: hyperpolarizability of embedded water molecules

Implementation of the electrostatic-map convolutional neural network (EM-CNN) described in **“Predicting the Molecular Hyperpolarizabilities of Embedded Water Molecules using Machine Learning Approaches”**, Guillaume Le Breton et al. The final journal reference will be added after publication.

EM-CNN predicts the first hyperpolarizability tensor of a water molecule from three-dimensional maps of the surrounding electric field (`Ex`, `Ey`, `Ez`). Maps and predictions use a molecular reference frame; separate transformations recover the laboratory-frame tensor for evaluation.

## Run the example

See **[USAGE.md](USAGE.md)** for installation, the required XYZ format, inputs/outputs, and all preparation and training settings. After installing the dependencies, run from the repository root:

```bash
python Example/prepare_data.py
python Example/run_cnn.py
```

The first command prepares the data on CPU. Run the second on a GPU machine or allocation to train the CNN and evaluate its best checkpoint. Both scripts expose their settings and named function calls; no cluster-specific submission script is required.

## Contents

- **`Example/`**: the small `1000-200` example, its XYZ/pickle data and rotation metadata, plus the two execution scripts. The actual XYZ counts are **800 training, 200 test, and 200 validation configurations**. Here `test` monitors training and `valid` is used for final evaluation.
- **`Scripts/`**: XYZ conversion, grid generation, normalization/augmentation, CNN training, and molecular/laboratory evaluation.
- **`USAGE.md`**: detailed installation and workflow guide.
- **`requirements.txt`**: Python dependencies; choose a suitable PyTorch GPU build as explained in the guide.

Generated data go to `Example/prepared/`; models, logs, and evaluation reports go to `Example/cnn_run/`. The scripts regenerate the intermediate files from the supplied XYZ inputs.

Additional data and the final trained `.keras` model will be made available very soon on Zenodo; the link and DOI will be added here. Small XYZ examples are already included in this repository.

## Contact

For questions, help running the code, or requests for additional data, [open an issue on GitHub](https://github.com/glb96/EM-CNN_water_hyperpola/issues) or contact [Guillaume Le Breton](https://github.com/glb96).

Email: **[address to be added]**.
