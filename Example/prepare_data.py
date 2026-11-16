#!/usr/bin/env python3
"""CPU stage: XYZ -> molecular data -> field maps -> normalized CNN datasets.

Run from any directory: python /path/to/repo/Example/prepare_data.py
Parameters below are the editable example configuration, not scheduler settings.
"""
import argparse
import copy
import json
import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Scripts'))

# 1. Input files and electrostatic model (charges in elementary-charge units).
INPUT_DIR = ROOT / 'Example'
OUTPUT_DIR = INPUT_DIR / 'prepared'
WATER_CHARGES = (-0.820, 0.410, 0.410)  # O, H1, H2; assigned at XYZ conversion.

# 2. Grid settings: lengths in angstroms; initial box is [-L, L]^3.
SMOOTHING = 0.6
MASK_EXTENSION = 2.4
BOX_HALF_WIDTH = 6.0
GRID_POINTS = 35  # Per axis, before cropping; spacing = 2L/(GRID_POINTS - 1).
WORKERS = 1      # Increase to suit the CPU allocation and available RAM.

# 3. Dataset preparation. Preserve original order; no random mixing here.
RATIO_DATA_TO_USE = 1.0
AUGMENT_X = True
AUGMENT_Y = True
MEMORY_SAVING = True
USE_R2_1 = False
USE_LOG_MEAN = False


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def prepare(*, input_dir, output_dir, water_charges, smoothing, mask_extension,
            box_half_width, grid_points, workers, ratio_data_to_use,
            augment_x, augment_y, memory_saving, use_r2_1, use_log_mean):
    import numpy as np
    from xyz_to_pickle_with_rotations import xyz_to_pickle
    from create_electrostatic_grid import generate_maps
    from prepare_data_set_CNN_new import prepare_data_for_GNN_ready_to_eat, normalization_stats
    from augment_validation_rotations import augment_rotations

    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not 0 < ratio_data_to_use <= 1:
        raise ValueError('ratio_data_to_use must be in (0, 1]')
    if use_log_mean:
        raise ValueError('This example keeps use_log_mean=False: its scale is split-dependent')
    for split in ('train', 'test', 'valid'):
        if not (input_dir / f'{split}.xyz').is_file():
            raise FileNotFoundError(input_dir / f'{split}.xyz')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'{output_dir} is not empty; choose a new --output-dir')
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = None
    stats = copy.deepcopy(normalization_stats)
    # Fixed beta constants, field means/stds fitted on training maps only.
    counts, grid_shape = {}, None
    for split in ('train', 'test', 'valid'):
        print(f'\n=== {split}: XYZ conversion, grid, CNN preparation ===', flush=True)
        work = output_dir / split
        work.mkdir()
        raw_path = work / f'{split}.p'
        raw_rotations = work / 'valid_rot.p' if split == 'valid' else None
        xyz_to_pickle(
            input_xyz=str(input_dir / f'{split}.xyz'), output_p=str(raw_path),
            rotation_output=str(raw_rotations) if raw_rotations else None,
            water_charges=water_charges,
        )
        with raw_path.open('rb') as handle:
            raw = pickle.load(handle)
        counts[split] = raw['T_steps'] * raw['N_molecule']
        selected = int(counts[split] * ratio_data_to_use)
        if selected < (2 if split == 'valid' else 1):
            raise ValueError(f'Too few selected samples in {split}')
        if reference is None:
            reference = copy.deepcopy(raw[(0, 0)])
        del raw
        maps_path = work / f'{split}_grid.p'
        # Grid visualizations and temporary memmaps stay inside this split's directory.
        with working_directory(work):
            shape = generate_maps(
                input_path=raw_path, output_path=maps_path, grid_points=grid_points,
                box_half_width=box_half_width, mask_extension=mask_extension,
                smoothing=smoothing, workers=workers, reference_configuration=reference,
            )
            if min(shape) < 8:
                raise ValueError('Cropped grid too small for three average-pooling layers')
            if grid_shape is not None and shape != grid_shape:
                raise ValueError('Grid dimensions differ between splits')
            grid_shape = shape
            used_stats = prepare_data_for_GNN_ready_to_eat(
                name_data_for_training=str(maps_path),
                name_data_set=str(output_dir / f'{split}_CNN.p'),
                ratio_data_to_use=ratio_data_to_use, memory_saving=memory_saving,
                use_r2_1=use_r2_1, use_log_mean=use_log_mean,
                do_sym_augmentation=augment_x or augment_y,
                aug_along_x=augment_x, aug_along_y=augment_y,
                normalization_stats=copy.deepcopy(stats), print_stats=True,
            )
        if split == 'train':
            stats = used_stats
        else:
            for key in stats:
                np.testing.assert_allclose(used_stats[key], stats[key], rtol=0, atol=0)
        if split == 'valid':
            with raw_rotations.open('rb') as handle:
                rotations = pickle.load(handle)
            augmentation = ('x' if augment_x else '') + ('y' if augment_y else '')
            rotations = augment_rotations(
                data=rotations, augmentation=augmentation or 'none', ratio=ratio_data_to_use)
            with (output_dir / 'valid_rot_aug.p').open('wb') as handle:
                pickle.dump(rotations, handle)

    # Keep an exact copy of the beta constants for standalone laboratory evaluation.
    (output_dir / 'beta_reference.py').write_text(
        'import numpy as np\n'
        f'beta_mean_ref = np.array({np.asarray(stats["beta_means"]).tolist()!r})\n'
        f'beta_std_ref = np.array({np.asarray(stats["beta_stds"]).tolist()!r})\n',
        encoding='utf-8',
    )
    manifest = dict(
        input_dir=str(input_dir), counts_original=counts,
        counts_prepared={s: int(n * ratio_data_to_use) * (2 if augment_x else 1)
                         * (2 if augment_y else 1) for s, n in counts.items()},
        water_charges=list(water_charges), smoothing=smoothing,
        mask_extension=mask_extension, box_half_width=box_half_width,
        grid_points=grid_points, cropped_grid_shape=list(grid_shape), workers=workers,
        ratio_data_to_use=ratio_data_to_use, augment_x=augment_x, augment_y=augment_y,
        use_r2_1=use_r2_1, use_log_mean=use_log_mean, shuffled=False,
        normalization={k: np.asarray(v).tolist() for k, v in stats.items()},
    )
    (output_dir / 'preparation.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nPreparation complete: {output_dir}')
    print('Original sample counts:', counts)
    print('Next: python Example/run_cnn.py --prepared-dir "' + str(output_dir) + '"')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--smoothing', type=float, default=SMOOTHING)
    parser.add_argument('--mask-extension', type=float, default=MASK_EXTENSION)
    parser.add_argument('--grid-points', type=int, default=GRID_POINTS)
    parser.add_argument('--box-half-width', type=float, default=BOX_HALF_WIDTH)
    args = parser.parse_args()
    prepare(
        input_dir=args.input_dir, output_dir=args.output_dir,
        water_charges=WATER_CHARGES, smoothing=args.smoothing,
        mask_extension=args.mask_extension, box_half_width=args.box_half_width,
        grid_points=args.grid_points, workers=args.workers,
        ratio_data_to_use=RATIO_DATA_TO_USE, augment_x=AUGMENT_X, augment_y=AUGMENT_Y,
        memory_saving=MEMORY_SAVING, use_r2_1=USE_R2_1, use_log_mean=USE_LOG_MEAN,
    )


if __name__ == '__main__':
    main()
