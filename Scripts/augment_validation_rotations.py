#!/usr/bin/env python3
"""Align valid_rot.p with the symmetry augmentation used in CNN preparation.
Default: python augment_validation_rotations.py valid_rot.p valid_rot_aug.p
Keep eval_labo.py beside this script. Only load trusted pickle files.
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
from eval_labo import check_matrices, expand_beta, flatten_beta, rotate


def augment_rotations(data, augmentation='xy', ratio=1.0):
    if data.get('format') != 'cnn_validation_rotations_v1' or data.get('stage') != 'raw':
        raise ValueError('Expected RAW rotations from xyz_to_pickle_with_rotations.py; '
                         'refusing repeated augmentation')
    if augmentation not in ('none', 'x', 'y', 'xy') or not 0 < ratio <= 1:
        raise ValueError('Invalid augmentation or ratio (must be in (0,1])')
    R = np.asarray(data['R_lab_to_mol'], dtype=np.float64)
    beta = np.asarray(data['beta_mol'], dtype=np.float32)
    ids = np.asarray(data['source_index'], dtype=np.int64)
    check_matrices(R)  # det=-1 is expected for the original molecular basis.
    if beta.shape != (len(R), 18) or ids.shape != (len(R), 2) or not np.isfinite(beta).all():
        raise ValueError('Inconsistent raw rotation metadata')
    n = int(len(R)*ratio)
    if n == 0:
        raise ValueError('Empty selection')
    R, beta, ids = R[:n], beta[:n], ids[:n]
    transforms = [('identity', [1, 1, 1])]
    if augmentation in ('x', 'xy'):
        transforms.append(('x', [-1, 1, 1]))
    if augmentation in ('y', 'xy'):
        transforms.append(('y', [1, -1, 1]))
    if augmentation == 'xy':
        transforms.append(('xy', [-1, -1, 1]))
    count = len(transforms)
    R_aug = np.empty((n, count, 3, 3), dtype=np.float64)
    beta_aug = np.empty((n, count, 18), dtype=np.float32)
    full = expand_beta(beta)
    for q, (_, signs) in enumerate(transforms):
        S = np.broadcast_to(np.diag(signs), R.shape)
        # v_aug = S @ v_mol = S @ R @ v_lab.
        R_aug[:, q] = S @ R
        beta_aug[:, q] = flatten_beta(rotate(full, S))
    R_aug = R_aug.reshape(-1, 3, 3)
    result = dict(data)
    result.update(stage='augmented', augmentation=augmentation, ratio=ratio,
                  n_original=n, n_samples=n*count,
                  convention='v_aug = R_lab_to_mol @ v_lab; v_lab = to_lab @ v_aug',
                  R_lab_to_mol=R_aug, to_lab=R_aug.transpose(0, 2, 1).copy(),
                  beta_mol=beta_aug.reshape(-1, 18),
                  source_index=np.repeat(ids, count, axis=0),
                  variant=np.tile([name for name, _ in transforms], n))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', nargs='?', default='valid_rot.p')
    p.add_argument('output', nargs='?', default='valid_rot_aug.p')
    p.add_argument('--augmentation', choices=['xy', 'x', 'y', 'none'], default='xy')
    p.add_argument('--ratio', type=float, default=1.)
    args = p.parse_args()
    with open(args.source, 'rb') as f:
        data = pickle.load(f)
    result = augment_rotations(data, args.augmentation, args.ratio)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement also permits explicitly using the same source/output path.
    import tempfile
    import os
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as f:
            tmp = f.name
            pickle.dump(result, f)
        os.replace(tmp, output)
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    print(f'Saved {output}: {result["n_original"]} originals, {result["n_samples"]} rows')
    print('Order within each configuration:', ', '.join(dict.fromkeys(result['variant'])))


if __name__ == '__main__':
    main()
