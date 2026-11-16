#!/usr/bin/env python3
"""Convert extxyz to the existing molecular dataset; optionally save rotations.

Example:
    python xyz_to_pickle_with_rotations.py valid.xyz valid.p --rotations valid_rot.p

Requires NumPy. Coordinates are in angstroms; l3_tensor is the laboratory-frame
beta tensor in the units used by the downstream model. Every water must be in
O,H,H order. Fixed charges are (-0.820, 0.410, 0.410); the added field is zero.
No change to training data format or molecular-frame convention.
"""
import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Any, Optional

import numpy as np


def iter_extxyz_frames(path: str) -> Iterator[Tuple[int, str, List[str]]]:
    with open(path, 'r') as f:
        while True:
            nat_line = f.readline()
            if not nat_line:
                break
            if not nat_line.strip():
                continue
            natoms = int(nat_line.strip())
            if natoms < 0:
                raise ValueError('Negative atom count')
            header = f.readline()
            if not header:
                raise ValueError('Truncated XYZ: missing header')
            atom_lines = []
            for _ in range(natoms):
                line = f.readline()
                if not line:
                    raise ValueError('Truncated XYZ: missing atom line')
                atom_lines.append(line.rstrip('\n'))
            yield natoms, header.rstrip('\n'), atom_lines


def parse_tensor_from_header(header: str, key: str) -> np.ndarray:
    m = re.search(r'\b' + re.escape(key) + r'="_JSON\s+(.+?)"(?=\s|$)', header)
    if m is None:
        raise ValueError(f'Could not find {key}="_JSON ..." in header line.')
    beta = np.array(json.loads(m.group(1)), dtype=np.float64)
    if beta.shape != (3, 3, 3) or not np.isfinite(beta).all():
        raise ValueError(f'Expected finite beta of shape (3,3,3), got {beta.shape}')
    return beta


def parse_hyperpolarizability_from_header(header: str) -> np.ndarray:
    return parse_tensor_from_header(header, 'hyperpolarizability')


def parse_l3_tensor_from_header(header: str) -> np.ndarray:
    return parse_tensor_from_header(header, 'l3_tensor')


def parse_atom_line(line: str) -> Tuple[str, np.ndarray]:
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f'Bad atom line: {line}')
    xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError(f'Non-finite coordinates: {line}')
    return parts[0], xyz


def build_molecular_frame(O: np.ndarray, H1: np.ndarray, H2: np.ndarray) -> np.ndarray:
    """EXACT original convention: v_mol = R @ v_lab, det(R)=-1.

    This orthogonal basis is left-handed. Deliberately preserve it because
    existing grids, targets and trained models use this convention.
    """
    x = H2 - H1
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-12:
        raise ValueError('H1 and H2 are too close; cannot define HH axis.')
    ex = x / x_norm
    mid = 0.5 * (H1 + H2)
    z0 = mid - O
    z0_norm = np.linalg.norm(z0)
    if z0_norm < 1e-12:
        raise ValueError('Bisector vector is too small.')
    ez0 = z0 / z0_norm
    y = np.cross(ex, ez0)
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-12:
        raise ValueError('Cannot define out-of-plane axis: near-linear geometry.')
    ey = y / y_norm
    ez = np.cross(ey, ex)
    B = np.column_stack([ex, ey, ez])
    return B.T


def rotate_positions(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (R @ pos.T).T


def rotate_beta(beta: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.einsum('ip,jq,kr,pqr->ijk', R, R, R, beta)


def _clean(x):
    return np.asarray(x, dtype=np.float64)


def xyz_to_pickle(input_xyz: str, output_p: Optional[str],
                  rotation_output: Optional[str] = None,
                  water_charges=(-0.820, 0.410, 0.410)) -> None:
    """output_p=None permits rotation-only recovery for an existing dataset.

    rotation_output=None (default): original behavior, no rotation file.
    rotation_output='valid_rot.p': also save a small separate dictionary.
    The XYZ must be the SAME validation file, in the SAME frame/atom order.
    """
    paths = [Path(x).resolve() for x in [input_xyz, output_p, rotation_output] if x is not None]
    if len(paths) != len(set(paths)):
        raise ValueError('Input XYZ, dataset and rotation output must have distinct paths')
    if output_p is None and rotation_output is None:
        raise ValueError('No output requested')
    water_charges = np.asarray(water_charges, dtype=np.float64)
    if water_charges.shape != (3,) or not np.isfinite(water_charges).all():
        raise ValueError("water_charges must contain three finite O,H,H charges")
    all_conformations: Dict[Any, Any] = {}
    matrices, betas = [], []
    T_steps = 0
    for t, (nat, header, atom_lines) in enumerate(iter_extxyz_frames(input_xyz)):
        beta_lab = parse_l3_tensor_from_header(header)
        symbols, coords = [], []
        for line in atom_lines:
            s, xyz = parse_atom_line(line)
            symbols.append(s)
            coords.append(xyz)
        coords = np.array(coords, dtype=np.float64)
        if nat < 6:
            raise ValueError(f'Frame {t}: need at least 6 atoms, got {nat}')
        if symbols[:3] != ['O', 'H', 'H']:
            raise ValueError(f'Frame {t}: expected interest molecule in O,H,H order')
        if (nat - 3) % 3:
            raise ValueError(f'Frame {t}: environment atom count must be a multiple of 3')
        # Charges are assigned by position within each triplet: verify its order.
        for start in range(3, nat, 3):
            if symbols[start:start + 3] != ['O', 'H', 'H']:
                raise ValueError(
                    f'Frame {t}: expected environment molecule {start // 3} '
                    'in O,H,H order'
                )
        O_lab = coords[0].copy()
        coords_trans = coords - O_lab[None, :]
        R = build_molecular_frame(coords_trans[0], coords_trans[1], coords_trans[2])
        beta_mol = rotate_beta(beta_lab, R)
        if output_p is not None:
            coords_mol = rotate_positions(coords_trans, R)
            env_positions = coords_mol[3:, :].reshape(-1, 3, 3)
            env_charges = np.tile(water_charges,
                                  (len(env_positions), 1))
            all_conformations[(t, 0)] = {
                'pos_mol_interest': _clean(coords_mol[:3, :].copy()),
                'env_positions': _clean(env_positions),
                'env_charges': _clean(env_charges),
                'field_value': _clean([0., 0., 0.]),
                'beta': _clean(beta_mol),
                'atoms': ['O', 'H1', 'H2'],
            }
        if rotation_output is not None:
            matrices.append(R)
            # Same upper triangle and float32 cast as grid generation.
            betas.append(np.array([beta_mol[i, j, k] for i in range(3)
                                   for j in range(3) for k in range(j, 3)], dtype=np.float32))
        T_steps += 1
    if T_steps == 0:
        raise ValueError('Empty XYZ file')
    if output_p is not None:
        all_conformations['T_steps'] = T_steps
        all_conformations['N_molecule'] = 1
        Path(output_p).parent.mkdir(parents=True, exist_ok=True)
        with open(output_p, 'wb') as f:
            pickle.dump(all_conformations, f)
        print(f'Wrote: {output_p}')
    if rotation_output is not None:
        metadata = {
            'format': 'cnn_validation_rotations_v1', 'stage': 'raw',
            'convention': 'v_mol = R_lab_to_mol @ v_lab',
            'source_xyz': str(Path(input_xyz).resolve()),
            'T_steps': T_steps, 'N_molecule': 1,
            'R_lab_to_mol': np.stack(matrices),
            'beta_mol': np.stack(betas),
            'source_index': np.column_stack([np.arange(T_steps), np.zeros(T_steps, dtype=int)]),
        }
        Path(rotation_output).parent.mkdir(parents=True, exist_ok=True)
        with open(rotation_output, 'wb') as f:
            pickle.dump(metadata, f)
        print(f'Wrote: {rotation_output} ({T_steps} original matrices)')
    print(f'T_steps={T_steps}, N_molecule=1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_xyz', help='Multi-frame XYZ with l3_tensor headers')
    parser.add_argument('output_p', nargs='?', help='Output dataset; omit for rotation-only recovery')
    parser.add_argument('--rotations', metavar='FILE', help='Optional raw rotation metadata output')
    args = parser.parse_args()
    if args.output_p is None and args.rotations is None:
        parser.error('Provide output_p and/or --rotations')
    xyz_to_pickle(args.input_xyz, args.output_p, args.rotations)


if __name__ == '__main__':
    main()
