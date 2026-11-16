#!/usr/bin/env python3
"""GPU stage: train EM-CNN, load its best checkpoint, evaluate held-out data.

Run after prepare_data.py, on a machine/allocation with a CUDA-enabled PyTorch.
The --allow-cpu option is only intended for small execution checks.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Scripts'))
PREPARED_DIR = ROOT / 'Example' / 'prepared'
RUN_DIR = ROOT / 'Example' / 'cnn_run'

# Original training schedule: phase maxima, mini-batch sizes, learning rates.
SCHEDULE = [
    {'epochs': 20, 'batch_size': 256, 'lr': 1e-3},
    {'epochs': 15, 'batch_size': 64, 'lr': 5e-4},
    {'epochs': 30, 'batch_size': 128, 'lr': 2.5e-4},
    {'epochs': 30, 'batch_size': 64, 'lr': 1e-4},
    {'epochs': 75, 'batch_size': 128, 'lr': 1e-4},
]
SHUFFLE_TRAIN = False  # Match the original script; validation is never shuffled.
SEED = 42
EVALUATION_BATCH_SIZE = 128
EVALUATION_SAMPLES = 'all'  # Or 'original' to exclude augmented validation rows.


def run(*, prepared_dir, run_dir, allow_cpu=False, smoke_test=False, evaluate_only=False):
    os.environ['KERAS_BACKEND'] = 'torch'
    import torch
    from train_CNN_LR_batch import train_model
    from eval_labo import main as evaluate

    prepared_dir, run_dir = Path(prepared_dir).resolve(), Path(run_dir).resolve()
    for filename in ('train_CNN.p', 'test_CNN.p', 'valid_CNN.p',
                     'valid_rot_aug.p', 'beta_reference.py', 'preparation.json'):
        if not (prepared_dir / filename).is_file():
            raise FileNotFoundError(f'Missing {prepared_dir / filename}; run prepare_data.py first')
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError('No CUDA GPU detected. Use a GPU allocation or --allow-cpu for a small check.')
    print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    schedule = [dict(epochs=1, batch_size=8, lr=1e-3)] if smoke_test else SCHEDULE
    if not evaluate_only:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f'{run_dir} is not empty; choose a new --run-dir or --evaluate-only')
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'preparation.json').write_bytes((prepared_dir / 'preparation.json').read_bytes())
        (run_dir / 'training.json').write_text(json.dumps(dict(
            schedule=schedule, shuffle_train=SHUFFLE_TRAIN, seed=SEED,
            evaluation_samples=EVALUATION_SAMPLES, smoke_test=smoke_test,
            torch_version=torch.__version__,
        ), indent=2), encoding='utf-8')
        best_model = train_model(
            train_path=prepared_dir / 'train_CNN.p', test_path=prepared_dir / 'test_CNN.p',
            run_dir=run_dir, schedule=schedule, shuffle_train=SHUFFLE_TRAIN, seed=SEED,
            clipnorm=1.0, reduce_lr_factor=0.5, reduce_lr_patience=5, min_lr=1e-6,
            early_stopping_patience=15, save_each_epoch=True,
        )
    else:
        best_model = run_dir / 'models' / 'best-model.keras'
    if not best_model.is_file():
        raise FileNotFoundError(best_model)
    # Evaluation loads the selected checkpoint, not the last epoch's weights.
    evaluate(argv=[
        '--model', str(best_model),
        '--dataset', str(prepared_dir / 'valid_CNN.p'),
        '--rotations', str(prepared_dir / 'valid_rot_aug.p'),
        '--eval-reference', str(prepared_dir / 'beta_reference.py'),
        '--batch-size', str(EVALUATION_BATCH_SIZE),
        '--samples', EVALUATION_SAMPLES,
        '--output', str(run_dir / 'evaluation' / 'validation_labo'),
    ])
    print(f'\nTraining/evaluation results: {run_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=PREPARED_DIR)
    parser.add_argument('--run-dir', type=Path, default=RUN_DIR)
    parser.add_argument('--allow-cpu', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='One epoch only; not a benchmark run')
    parser.add_argument('--evaluate-only', action='store_true', help='Re-evaluate existing best checkpoint')
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
