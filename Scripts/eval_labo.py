#!/usr/bin/env python3
"""Evaluate the existing CNN in molecular and laboratory frames (no training).
Normalization is read from beta_mean_ref and beta_std_ref in the eval.py
beside this script, without executing eval.py. No normalization_stats.p is used.
Compares ML with the saved constant beta_mean_ref in both frames.
Also reports mean_ijk(RMSE_ijk / sigma_ijk) in the laboratory frame,
using reference standard deviations (ddof=0), for 18 and 27 components.
Only load trusted pickle / torch files.
"""
import argparse
import ast
import csv
import json
import os
import pickle
from pathlib import Path
import numpy as np

# Same order as flatten_beta in create_electrostatic_grid.py; only j<->k symmetry.
INDICES = [(i, j, k) for i in range(3) for j in range(3) for k in range(j, 3)]
LABELS = [''.join('xyz'[a] for a in ijk) for ijk in INDICES]


def expand_beta(beta):
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape[-1] != 18:
        raise ValueError('Expected 18 components')
    out = np.zeros(beta.shape[:-1] + (3, 3, 3), dtype=np.float64)
    for q, (i, j, k) in enumerate(INDICES):
        out[..., i, j, k] = out[..., i, k, j] = beta[..., q]
    return out


def flatten_beta(beta):
    return np.stack([beta[..., i, j, k] for i, j, k in INDICES], axis=-1)


def rotate(beta, matrix):
    """Column vectors: v_out = matrix @ v_in. Works also for reflections."""
    return np.einsum('nia,njb,nkc,nabc->nijk', matrix, matrix, matrix,
                     beta, optimize=True)


def check_matrices(matrix, proper=False):
    if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
        raise ValueError('Matrices must have shape (N,3,3)')
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite matrix')
    if not np.allclose(matrix @ matrix.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0):
        raise ValueError('Non-orthogonal matrix')
    if proper and not np.allclose(np.linalg.det(matrix), 1., atol=1e-5, rtol=0):
        raise ValueError('Source matrices must be proper rotations (det=+1)')


def metrics(true, pred):
    """Component-centered SST, as in eval.py. Undefined R2 is NaN."""
    if len(true) < 2 or not (np.isfinite(true).all() and np.isfinite(pred).all()):
        raise ValueError('Need >=2 samples and finite predictions/targets')
    diff = true - pred
    sse = np.sum(diff**2, axis=0)
    sst = np.sum((true - true.mean(axis=0))**2, axis=0)
    r2 = np.full(sst.shape, np.nan)
    np.divide(sse, sst, out=r2, where=sst > 0)
    return dict(r2_global=float(1-sse.sum()/sst.sum()) if sst.sum() > 0 else float('nan'),
                mse_total=float(np.mean(diff**2)), mae_global=float(np.mean(abs(diff))),
                r2_per=1-r2, mse_per=np.mean(diff**2, axis=0),
                mae_per=np.mean(abs(diff), axis=0), sse=float(sse.sum()), sst=float(sst.sum()))


def normalized_rmse(true, pred):
    """Unweighted component mean of RMSE/sigma; sigma from physical references.

    Both RMSE and sigma use division by N (ddof=0). Thus each ratio equals
    sqrt(1 - R2_component), but their mean is not 1 - R2_global.
    A zero reference sigma makes that ratio and the full mean undefined (NaN);
    do not silently drop components or add an arbitrary epsilon.
    """
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if (true.ndim != 2 or true.shape != pred.shape or len(true) < 2
            or not np.isfinite(true).all() or not np.isfinite(pred).all()):
        raise ValueError('Expected matching finite arrays (N, components), N >= 2')
    rmse = np.sqrt(np.mean((true - pred)**2, axis=0))
    sigma = np.std(true, axis=0, ddof=0)
    ratio = np.full(sigma.shape, np.nan, dtype=np.float64)
    np.divide(rmse, sigma, out=ratio, where=sigma > 0)
    return dict(rmse_per=rmse, sigma_per=sigma, sigma_ddof=0,
                rmse_over_sigma_per=ratio, rmse_over_sigma_mean=float(np.mean(ratio)),
                zero_sigma_components=np.flatnonzero(sigma == 0))


def check_targets(actual, expected):
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=2e-5, rtol=2e-6):
        raise ValueError('Validation targets do not match rotation metadata: check source valid.p, '
                         'sample order, augmentation, ratio and fixed beta constants in eval.py.')


def load_rotations(path):
    """Read the augmented .p file; accept det=+1 and det=-1 bases."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if data.get('format') != 'cnn_validation_rotations_v1' or data.get('stage') != 'augmented':
        raise ValueError('Expected valid_rot_aug.p from augment_validation_rotations.py, '
                         'not the raw valid_rot.p file')
    matrices = np.asarray(data['to_lab'], dtype=np.float64)
    forward = np.asarray(data['R_lab_to_mol'], dtype=np.float64)
    expected = np.asarray(data['beta_mol'], dtype=np.float64)
    variants = np.asarray(data['variant'])
    ids = np.asarray(data['source_index'], dtype=np.int64)
    check_matrices(matrices)
    n = len(matrices)
    if (forward.shape != matrices.shape or expected.shape != (n, 18)
            or variants.shape != (n,) or ids.shape != (n, 2)
            or not np.isfinite(expected).all()):
        raise ValueError('Inconsistent rotation metadata shapes / targets')
    if not np.allclose(forward.transpose(0, 2, 1), matrices, atol=1e-12, rtol=0):
        raise ValueError('Forward and inverse transforms disagree')
    if not np.isin(variants, ['identity', 'x', 'y', 'xy']).all():
        raise ValueError('Unknown augmentation variant')
    return matrices, expected, variants, ids


def evaluate_frames(true_mol, pred_mol, to_lab):
    """Score one predictor in both frames, using identical selected samples."""
    true_full, pred_full = expand_beta(true_mol), expand_beta(pred_mol)
    true_lab_full, pred_lab_full = rotate(true_full, to_lab), rotate(pred_full, to_lab)
    true_lab, pred_lab = flatten_beta(true_lab_full), flatten_beta(pred_lab_full)
    scores = {}
    for name, a, b in [
        ('molecular_18', true_mol, pred_mol), ('laboratory_18', true_lab, pred_lab),
        ('molecular_27', true_full.reshape(-1, 27), pred_full.reshape(-1, 27)),
        ('laboratory_27', true_lab_full.reshape(-1, 27), pred_lab_full.reshape(-1, 27)),
    ]:
        scores[name] = metrics(a, b)
        if name.startswith('laboratory_'):
            scores[name].update(normalized_rmse(a, b))
    if not np.isclose(scores['molecular_27']['sse'], scores['laboratory_27']['sse'],
                      rtol=1e-4, atol=1e-10):
        raise ValueError('Full tensor error norm is not conserved by rotation')
    return scores, true_lab, pred_lab


def load_beta_reference(path):
    """Read only the two literal np.array assignments, without importing eval.py.

    A normal import would also run the evaluation call at the end of eval.py.
    This reads the arrays as written in the supplied eval.py, without executing
    its imports, functions or model-loading code.
    """
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    values = {}
    names = ('beta_mean_ref', 'beta_std_ref')
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id not in names:
                continue
            if target.id in values:
                raise ValueError(f'Multiple definitions of {target.id} in {path}')
            call = node.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == 'np' and call.func.attr == 'array'
                    and len(call.args) == 1):
                raise ValueError(f'{target.id} must be defined as np.array([...]) in {path}')
            values[target.id] = np.asarray(ast.literal_eval(call.args[0]), dtype=np.float64)
    if any(name not in values for name in names):
        raise ValueError(f'Missing beta_mean_ref / beta_std_ref in {path}')
    mean, std = (values[name] for name in names)
    if (mean.shape != (18,) or std.shape != (18,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std <= 0)):
        raise ValueError(f'Invalid beta reference constants in {path}')
    return mean, std


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='./models/best-model.keras')
    p.add_argument('--dataset', default='../valid_CNN_small.p')
    p.add_argument('--rotations', default='valid_rot_aug.p', help='Augmented rotation pickle')
    p.add_argument('--eval-reference', default=str(Path(__file__).resolve().with_name('eval.py')),
                   help='File defining the fixed beta constants; default: eval.py beside this script')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--samples', choices=['all', 'original'], default='all')
    p.add_argument('--output', default='validation_labo')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    os.environ['KERAS_BACKEND'] = 'torch'
    import torch
    from torch.utils.data import DataLoader, Subset
    import keras

    reference_path = Path(args.eval_reference).resolve()
    mean, std = load_beta_reference(reference_path)
    print(f'Fixed beta normalization from: {reference_path}')
    matrices, expected, variants, ids = load_rotations(args.rotations)
    dataset = torch.load(args.dataset, map_location='cpu', weights_only=False)
    if len(dataset) != len(matrices) or expected.shape != (len(dataset), 18):
        raise ValueError('Dataset / rotation count mismatch')
    indices = np.arange(len(dataset)) if args.samples == 'all' else np.flatnonzero(variants == 'identity')
    if len(indices) < 2:
        raise ValueError('Need at least two selected validation samples')
    # Check every target before expensive prediction, including skipped augmented rows.
    for start in range(0, len(dataset), args.batch_size):
        stop = min(start + args.batch_size, len(dataset))
        y = torch.stack([dataset[i][1] for i in range(start, stop)]).cpu().numpy()
        check_targets(y.astype(np.float64)*std + mean, expected[start:stop])
    model = keras.models.load_model(args.model, compile=False)
    predictions, targets = [], []
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    for x, y in loader:
        pred = np.asarray(model.predict_on_batch(x.to(torch.float32)), dtype=np.float64)
        if pred.shape != (len(y), 18):
            raise ValueError(f'Unexpected prediction shape: {pred.shape}')
        predictions.append(pred*std + mean)
        targets.append(y.cpu().numpy().astype(np.float64)*std + mean)
    true_mol, pred_mol = np.concatenate(targets), np.concatenate(predictions)
    results, true_lab, pred_lab = evaluate_frames(true_mol, pred_mol, matrices[indices])
    # Baseline: the saved molecular mean from eval.py, identical for all samples.
    # No mean is recomputed from validation labels for the baseline prediction.
    constant_mol = np.broadcast_to(mean, true_mol.shape).copy()
    constant_scores, _, constant_lab = evaluate_frames(true_mol, constant_mol, matrices[indices])
    baselines = {
        'constant_reference': dict(beta_mol_constant=mean, metrics=constant_scores,
                                   uses_validation_labels=False)
    }
    baseline_arrays = dict(constant_reference_mol=constant_mol,
                           constant_reference_lab=constant_lab)
    print('\nconstant_reference = saved beta_mean_ref from eval.py (zero normalized output).')
    for predictor, scores in [('ML', results)] + [(k, v['metrics']) for k, v in baselines.items()]:
        print(f'\n===== {predictor} =====')
        for name, m in scores.items():
            print(f'{name}: R2={m["r2_global"]:.9f}  MSE={m["mse_total"]:.6e}  MAE={m["mae_global"]:.6e}')
            if name.startswith('laboratory_'):
                print(f'  Mean(RMSE_ijk/sigma_ijk) = {m["rmse_over_sigma_mean"]:.9f}'
                      f'  (reference sigma, ddof=0, {len(m["sigma_per"])} components)')
                if len(m['zero_sigma_components']):
                    print('  Undefined mean: zero sigma for component indices',
                          m['zero_sigma_components'].tolist())
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    def serial(value):
        if isinstance(value, dict):
            return {k: serial(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return serial(value.tolist())
        if isinstance(value, list):
            return [serial(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    report = dict(n_samples=len(indices), samples=args.samples, model=args.model,
                  dataset=args.dataset, rotations=args.rotations,
                  normalization=dict(source=str(reference_path), beta_mean_ref=mean, beta_std_ref=std),
                  metrics=results, baselines=baselines)
    Path(str(prefix)+'.json').write_text(json.dumps(serial(report), indent=2), encoding='utf-8')
    with open(str(prefix)+'_components.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'component', 'R2', 'MSE', 'MAE'])
        for frame in ['molecular_18', 'laboratory_18']:
            m = results[frame]
            for q, label in enumerate(LABELS):
                writer.writerow([frame, label, m['r2_per'][q], m['mse_per'][q], m['mae_per'][q]])
    with open(str(prefix)+'_baselines_components.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['baseline', 'frame', 'component', 'R2', 'MSE', 'MAE'])
        for predictor, baseline in baselines.items():
            for frame in ['molecular_18', 'laboratory_18']:
                m = baseline['metrics'][frame]
                for q, label in enumerate(LABELS):
                    writer.writerow([predictor, frame, label, m['r2_per'][q],
                                     m['mse_per'][q], m['mae_per'][q]])
    # Separate export keeps the previous CSV layouts unchanged.
    with open(str(prefix)+'_lab_normalized_rmse.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['predictor', 'frame', 'component', 'RMSE', 'sigma_ref_ddof0',
                         'RMSE_over_sigma'])
        labels_27 = [''.join('xyz'[a] for a in (i, j, k))
                     for i in range(3) for j in range(3) for k in range(3)]
        for predictor, scores in [('ML', results)] + [(k, v['metrics']) for k, v in baselines.items()]:
            for frame, labels in [('laboratory_18', LABELS), ('laboratory_27', labels_27)]:
                m = scores[frame]
                for q, label in enumerate(labels):
                    writer.writerow([predictor, frame, label, m['rmse_per'][q],
                                     m['sigma_per'][q], m['rmse_over_sigma_per'][q]])
                writer.writerow([predictor, frame, 'MEAN', '', '', m['rmse_over_sigma_mean']])
    np.savez_compressed(str(prefix)+'_predictions.npz', true_mol=true_mol, pred_mol=pred_mol,
                        true_lab=true_lab, pred_lab=pred_lab, dataset_index=indices,
                        source_index=ids[indices], variant=variants[indices], labels=LABELS, **baseline_arrays)
    print(f'Saved {prefix}.json, {prefix}_components.csv, '
          f'{prefix}_baselines_components.csv, {prefix}_lab_normalized_rmse.csv, '
          f'{prefix}_predictions.npz')


if __name__ == '__main__':
    main()
