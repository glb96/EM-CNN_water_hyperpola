import numpy as np

import sys
import os 
import multiprocessing 
import copy 

import pickle
import torch

import argparse

smoothing_factor = 0.6
size_atom = 2.4
size_grid = 35


# Constants for conversion
e_charge = 1.602e-19  # Coulombs (elementary charge in Coulombs)
epsilon_0 = 8.854e-12  # Vacuum permittivity in C^2 / (N·m^2) (SI units)
bohr_to_angstrom = 0.529177  # 1 Bohr radius in Angstroms
atomic_unit_to_volts = 27.2114  # 1 a.u. in eV

###########################################################################
###########################################################################
###########################################################################


def check_mask_boundaries(maskgrid):
    """
    Check if each boundary plane (at min/max x, y, z) is completely filled with zeros.
    If any boundary plane contains a value of 1, raise an exception.
    """
    # Extract the boundary planes
    x_min_plane = maskgrid[0, :, :]    # First X-plane
    x_max_plane = maskgrid[-1, :, :]   # Last X-plane
    y_min_plane = maskgrid[:, 0, :]    # First Y-plane
    y_max_plane = maskgrid[:, -1, :]   # Last Y-plane
    z_min_plane = maskgrid[:, :, 0]    # First Z-plane
    z_max_plane = maskgrid[:, :, -1]   # Last Z-plane

    # Check if any of these planes contain ones
    if np.max(x_min_plane) == 1:
        raise Exception("ERROR: The minimal X-plane is not filled with zeros! Decrease your minimal x value for the grid or decrease the size per atom!")
    if np.max(x_max_plane) == 1:
        raise Exception("ERROR: The maximal X-plane is not filled with zeros! Increase your maximal x value for the grid or decrease the size per atom!")
    if np.max(y_min_plane) == 1:
        raise Exception("ERROR: The minimal Y-plane is not filled with zeros! Decrease your minimal y value for the grid or decrease the size per atom!")
    if np.max(y_max_plane) == 1:
        raise Exception("ERROR: The maximal Y-plane is not filled with zeros! Increase your maximal y value for the grid or decrease the size per atom!")
    if np.max(z_min_plane) == 1:
        raise Exception("ERROR: The minimal Z-plane is not filled with zeros! Decrease your minimal z value for the grid or decrease the size per atom!")
    if np.max(z_max_plane) == 1:
        raise Exception("ERROR: The maximal Z-plane is not filled with zeros! Increase your maximal z value for the grid or decrease the size per atom!")


    print("Check passed: Each boundary plane contains only zeros.")

def crop_grid_to_minimal_shape(X, Y, Z, maskgrid):
    """
    Crop the X, Y, Z grid and maskgrid to the minimal bounding box while 
    ensuring that each boundary plane still contains at least one zero.
    """

    # Find indices where maskgrid == 1
    x_indices, y_indices, z_indices = np.where(maskgrid == 1)

    # Find the bounding box with extra padding for boundary zeros
    x_min, x_max = max(0, np.min(x_indices) - 1), min(maskgrid.shape[0] - 1, np.max(x_indices) + 1)
    y_min, y_max = max(0, np.min(y_indices) - 1), min(maskgrid.shape[1] - 1, np.max(y_indices) + 1)
    z_min, z_max = max(0, np.min(z_indices) - 1), min(maskgrid.shape[2] - 1, np.max(z_indices) + 1)

    # Crop the arrays
    maskgrid_cropped = maskgrid[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    X_cropped = X[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    Y_cropped = Y[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    Z_cropped = Z[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    # Perform the boundary check again
    check_mask_boundaries(maskgrid_cropped)

    print("Cropping successful: The grid has been resized to minimal shape while keeping boundary zeros.")
    
    return X_cropped, Y_cropped, Z_cropped, maskgrid_cropped

def refine_grid_according_to_atomic_position(atom_treshold, pos_mol_interest, X, Y, Z):
    maskgrid = np.zeros((N_grid[0], N_grid[1], N_grid[2]))
    for i in range(len(atom_treshold)):
        threshold = atom_treshold[i]
        pos = pos_mol_interest[i]
        for x_idx in range(N_grid[0]):
            for y_idx in range(N_grid[1]):
                for z_idx in range(N_grid[2]):
                    r_vec = np.array([X[x_idx, y_idx, z_idx], Y[x_idx, y_idx, z_idx], Z[x_idx, y_idx, z_idx]]) - pos
                    if np.linalg.norm(r_vec) < threshold:
                        maskgrid[x_idx, y_idx, z_idx] = 1

    # remove the unnecessary zeros: 
    check_mask_boundaries(maskgrid)
    X_cropped, Y_cropped, Z_cropped, maskgrid_cropped = crop_grid_to_minimal_shape(X, Y, Z, maskgrid)
    check_mask_boundaries(maskgrid_cropped)
    print('Old N_bin:', np.shape(X), 'New ones:', np.shape(X_cropped))
    return( X_cropped, Y_cropped, Z_cropped, maskgrid_cropped)
                

    
def create_grid(N_grid, L_bond_grid):
    # Create grid positions for each axis based on provided min, max, and N_grid values
    x_min, x_max = L_bond_grid[0]
    y_min, y_max = L_bond_grid[1]
    z_min, z_max = L_bond_grid[2]

    # Create arrays for the grid positions
    x_positions = np.linspace(x_min, x_max, N_grid[0])
    y_positions = np.linspace(y_min, y_max, N_grid[1])
    z_positions = np.linspace(z_min, z_max, N_grid[2])
    X, Y, Z =  np.meshgrid(x_positions, y_positions, z_positions, indexing='ij')
    return(X, Y, Z) 

def create_grid_for_water(LL_conformation,  N_grid, L_bond_grid, L_size_atom):
    # initialization of the grid     
    pos_mol_interest = LL_conformation['pos_mol_interest']
    atoms = LL_conformation['atoms']
    atom_treshold = []
    for i in range(len(atoms)):
        myatom = atoms[i]
        if myatom=='H1' or myatom=='H2':
            atom_treshold.append(L_size_atom[myatom])
        elif myatom == 'O':
            atom_treshold.append(L_size_atom[myatom])
        else:
            raise Exception('ERROR!!!!')
    mean_position = np.mean(pos_mol_interest, axis=0)
    pos_mol_interest = pos_mol_interest.tolist()  
    pos_mol_interest.append(mean_position)
    atoms.append('X')
    atom_treshold.append(L_size_atom['mean'])
    print(atom_treshold)
    
    X, Y, Z = create_grid(N_grid, L_bond_grid)
    X, Y, Z, maskgrid = refine_grid_according_to_atomic_position(atom_treshold, pos_mol_interest, X, Y, Z) 
    N_grid = np.shape(X) 
    save_xyz("molecule_interest.xyz", atoms, pos_mol_interest)
    # Save grid points (original and masked)
    save_grid_xyz("grid", X, Y, Z, maskgrid)
    return(N_grid, X, Y, Z, maskgrid)

def save_xyz(filename, atoms, positions):
    """
    Save an XYZ file given atom names and positions.
    
    Parameters:
        filename (str): Name of the output .xyz file.
        atoms (list of str): List of atomic symbols (e.g., ["O", "H", "H"]).
        positions (np.array): Nx3 array of atomic positions.
    """
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("XYZ file generated for VMD visualization\n")
        for atom, pos in zip(atoms, positions):
            f.write(f"{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")


def save_grid_xyz(grid_filename, X, Y, Z, maskgrid=None):
    """
    Save grid points as an XYZ file, using He for original grid and Ne for masked grid.
    
    Parameters:
        grid_filename (str): Base filename (e.g., "grid").
        X, Y, Z (np.array): Meshgrid arrays (shape: (N_x, N_y, N_z)).
        maskgrid (np.array, optional): Mask grid, same shape as (N_x, N_y, N_z).
    """
    all_positions = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Save the original grid (He atoms)
    save_xyz(f"{grid_filename}_original.xyz", ["He"] * len(all_positions), all_positions)
    
    # Save the masked grid (Ne atoms)
    if maskgrid is not None:
        masked_positions = all_positions[maskgrid.ravel() > 0]
        save_xyz(f"{grid_filename}_masked.xyz", ["Ne"] * len(masked_positions), masked_positions)



def built_electrostatic_potential(X, Y, Z, env_positions_mol, env_charges, field_value_mol=None, maskgrid=None, L_size_atom=None, TTT=-1, n=-1):
    """
    Build the electrostatic potential field for the given molecular environment.
    
    Parameters:
        N_grid: A tuple with the number of grid points for each axis (N_x, N_y, N_z).
        L_bond_grid: A tuple with the (min, max) positions for each axis ((min_x, max_x), (min_y, max_y), (min_z, max_z)).
        env_positions_mol: Molecular positions in the grid frame (in Angstroms).
        env_charges: The charges associated with the environment (in electron charge units).
        field_value_mol: Optional field value to be added to the whole grid (in atomic units).

    Returns:
        my_grid: 3D grid of the electrostatic field in atomic units, with shape (3, N_x, N_y, N_z).
    """
    # Initialize the electrostatic field grid (3 channels for the x, y, and z components)
    N_grid = np.shape(X)
    my_grid = np.zeros((3, N_grid[0], N_grid[1], N_grid[2]))

    if maskgrid is None:
        maskgrid = np.zeros((N_grid[0], N_grid[1], N_grid[2])) + 1 # all true
 
    if L_size_atom['max_wrt_grid'] is not None: 
        max_wrt_grid = L_size_atom['max_wrt_grid']
    else:
        max_wrt_grid = 0.05 # default value 

    # Add the contributions from the environment charges to the grid
    for n_neigh in range(len(env_charges)):
        L_charge = env_charges[n_neigh]
        for i in range(len(L_charge)):
            charge = L_charge[i]
            if np.absolute(charge) > 1e-5:
                pos = env_positions_mol[n_neigh,i,:]
                for x_idx in range(N_grid[0]):
                    for y_idx in range(N_grid[1]):
                        for z_idx in range(N_grid[2]):
                            if maskgrid[x_idx, y_idx, z_idx]:
                                r_vec = np.array([X[x_idx, y_idx, z_idx], Y[x_idx, y_idx, z_idx], Z[x_idx, y_idx, z_idx]]) - pos
                                r = np.linalg.norm(r_vec)  # Euclidean distance between grid point and charge position
                                field_contribution = charge / ((r**2+smoothing_factor**2)**(3/2)) * r_vec  # Coulomb field contribution
                                my_grid[:, x_idx, y_idx, z_idx] += field_contribution
    # Normalize the field with Coulomb's constant
    my_grid = my_grid*(bohr_to_angstrom**2) # for atomic units 
    # If a specific field value is provided, convert from atomic units to Volts and add it to the grid
    if field_value_mol is not None:
        for x_idx in range(N_grid[0]):
            for y_idx in range(N_grid[1]):
                for z_idx in range(N_grid[2]):
                    if maskgrid[x_idx, y_idx, z_idx]:
                        my_grid[:, x_idx, y_idx, z_idx] += field_value_mol
    return my_grid



def build_electrostatic_field_fast(X, Y, Z, env_positions_mol, env_charges, field_value_mol=None, maskgrid=None, L_size_atom=None, TTT=-1, n=-1):
    """
    Calcule E(r) = sum_i q_i * (r - r_i) / (|r-r_i|^2 + s^2)^(3/2)
    Retour: my_grid shape (3, Nx, Ny, Nz)
    """
    dtype = np.float32
    chunk_size=200_000
    Nx, Ny, Nz = X.shape

    if maskgrid is None:
        maskgrid = np.ones((Nx, Ny, Nz), dtype=bool)
    else:
        maskgrid = maskgrid.astype(bool, copy=False)

    # Flatten des coords sur les voxels "actifs" uniquement
    mask_flat = maskgrid.ravel()
    Xf = X.ravel()[mask_flat].astype(dtype, copy=False)
    Yf = Y.ravel()[mask_flat].astype(dtype, copy=False)
    Zf = Z.ravel()[mask_flat].astype(dtype, copy=False)

    coords = np.stack([Xf, Yf, Zf], axis=1)  # (Npts, 3)

    Npts = coords.shape[0]
    out_flat = np.zeros((Npts, 3), dtype=dtype)

    s2 = dtype(smoothing_factor * smoothing_factor)

    # Helper pour itérer sur toutes les charges sans créer de grosses listes
    def iter_charges():
        # env_charges[n_neigh] = liste/array de charges
        # env_positions_mol[n_neigh, i, :] = position correspondante
        for n_neigh in range(len(env_charges)):
            charges = np.asarray(env_charges[n_neigh])
            # positions (n_i, 3)
            pos = np.asarray(env_positions_mol[n_neigh], dtype=dtype)
            for i, q in enumerate(charges):
                q = float(q)
                if abs(q) > 1e-5:
                    yield q, pos[i]

    # Accumulation par charge, mais calcul vectorisé sur la grille (par chunks)
    for q, pos in iter_charges():
        pos = np.asarray(pos, dtype=dtype)

        for start in range(0, Npts, chunk_size):
            end = min(start + chunk_size, Npts)
            rv = coords[start:end] - pos[None, :]            # (chunk, 3)
            r2 = np.sum(rv * rv, axis=1) + s2               # (chunk,)
            inv = 1.0 / (r2 * np.sqrt(r2))                  # 1 / (r2)^(3/2)
            out_flat[start:end] += (q * inv)[:, None] * rv  # (chunk, 3)

    # Remettre en (3, Nx, Ny, Nz)
    my_grid = np.zeros((3, Nx * Ny * Nz), dtype=dtype)
    my_grid[:, mask_flat] = out_flat.T
    my_grid = my_grid.reshape(3, Nx, Ny, Nz)

    # Conversion unités
    my_grid *= (bohr_to_angstrom ** 2)

    # Ajout d’un champ uniforme (vectorisé)
    if field_value_mol is not None:
        fv = np.asarray(field_value_mol, dtype=dtype).reshape(3, 1)  # (3,1) ou (3,)
        my_grid[:, maskgrid] += fv

    return my_grid





def flatten_beta(my_beta, kleinman=False):
    if not kleinman:
        selected_beta = np.zeros(18)
        selected_beta[0] = my_beta[0, 0, 0]
        selected_beta[1] = my_beta[0, 0, 1]
        selected_beta[2] = my_beta[0, 0, 2]
        selected_beta[3] = my_beta[0, 1, 1]
        selected_beta[4] = my_beta[0, 1, 2]
        selected_beta[5] = my_beta[0, 2, 2]
        selected_beta[6] = my_beta[1, 0, 0]
        selected_beta[7] = my_beta[1, 0, 1]
        selected_beta[8] = my_beta[1, 0, 2]
        selected_beta[9] = my_beta[1, 1, 1]
        selected_beta[10] = my_beta[1, 1, 2]
        selected_beta[11] = my_beta[1, 2, 2]
        selected_beta[12] = my_beta[2, 0, 0]
        selected_beta[13] = my_beta[2, 0, 1]
        selected_beta[14] = my_beta[2, 0, 2]
        selected_beta[15] = my_beta[2, 1, 1]
        selected_beta[16] = my_beta[2, 1, 2]
        selected_beta[17] = my_beta[2, 2, 2]
    return(selected_beta)

###########################################################################
###########################################################################
###########################################################################

def process_t_n(args):
    TTT, n = args
    LL_conformation = _G["data"][(TTT, n)]

    X = _G["X"]; Y = _G["Y"]; Z = _G["Z"]
    maskgrid = _G["maskgrid"]
    L_size_atom = _G["L_size_atom"]

    field_value_mol = LL_conformation["field_value"]
    env_positions_mol = LL_conformation["env_positions"]
    env_charges = LL_conformation["env_charges"]
    
    # Build the electrostatic potential grid
    my_grid = build_electrostatic_field_fast(X, Y, Z, env_positions_mol, env_charges, field_value_mol=field_value_mol, maskgrid=maskgrid, L_size_atom=L_size_atom, TTT=TTT, n=n)
    
    # Get the corresponding beta values (assume these are 1D or 3D tensor)
    my_beta = LL_conformation['beta']  # Example: [beta_111, beta_112, ...] or 27 components
    selected_beta = flatten_beta(my_beta, kleinman=False)
    return(TTT, n, my_grid, selected_beta)

###########################################################################

# Globals for multiprocessing workers
_G = {}

def _init_worker(X, Y, Z, maskgrid, L_size_atom, LL_all_conformation):
    _G["X"] = X
    _G["Y"] = Y
    _G["Z"] = Z
    _G["maskgrid"] = maskgrid
    _G["L_size_atom"] = L_size_atom
    _G["data"] = LL_all_conformation
 
###########################################################################

def build_inputs_for_ML_parr(name_data_set, N_grid, L_bond_grid, L_size_atom, T_steps, N_molecule, num_processes):
    with open(name_data_set, "rb") as filetoload:
        LL_all_conformation = pickle.load(filetoload)

    if N_molecule == -1:
        N_molecule = LL_all_conformation["N_molecule"]
    else:
        if N_molecule > LL_all_conformation["N_molecule"]:
            raise Exception("N_molecule too large! max value:", LL_all_conformation["N_molecule"])

    if T_steps == -1:
        T_steps = LL_all_conformation["T_steps"]
    else:
        if T_steps > LL_all_conformation["T_steps"]:
            raise Exception("T_steps too large! max value:", LL_all_conformation["T_steps"])

    N_grid, X, Y, Z, maskgrid = create_grid_for_water(LL_all_conformation[(0, 0)], N_grid, L_bond_grid, L_size_atom)
    L_XYZ_grid = [X, Y, Z]

    N_config = T_steps * N_molecule
    L_fields = np.zeros((N_config, 3, N_grid[0], N_grid[1], N_grid[2]), dtype=np.float32)
    L_beta = np.zeros((N_config, 18), dtype=np.float32)

    all_args = [(TTT, n) for TTT in range(T_steps) for n in range(N_molecule)]

    # Pool avec initializer : X/Y/Z/maskgrid + dataset disponibles globalement dans chaque worker
    with multiprocessing.Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(X, Y, Z, maskgrid, L_size_atom, LL_all_conformation),
    ) as pool:
        results = pool.map(process_t_n, all_args)

    trotter = 0
    for TTT, n, my_grid, selected_beta in results:
        L_fields[trotter] = my_grid.astype(np.float32, copy=False)
        L_beta[trotter] = selected_beta.astype(np.float32, copy=False)
        trotter += 1

    fields_tensor = np.array(L_fields, copy=False)
    betas_tensor = np.array(L_beta, copy=False)

    return fields_tensor, betas_tensor, L_XYZ_grid    

########################################################################### 
###########################################################################
###########################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build electrostatic grids and save as pickle")
    parser.add_argument("name_data_set", type=str, help="Input pickle (.p) containing configurations")
    parser.add_argument("name_for_save", type=str, help="Output pickle (.p) for electrostatic maps")

    args = parser.parse_args()

    # Example usage
    T_steps = -1
    N_molecule = -1

    num_processes = 32

    name_data_set = args.name_data_set
    name_for_save = args.name_for_save

    # Example grid parameters
    N_grid = (int(size_grid), int(size_grid), int(size_grid))  # (N_x, N_y, N_z)
    L_bond_grid = ((-6, 6), (-6, 6), (-6, 6))
    L_size_atom = {}
    L_size_atom["H1"] = 0.8 + size_atom 
    L_size_atom["H2"] = 0.8 + size_atom
    L_size_atom["O"] = 1.0 + size_atom
    L_size_atom["mean"] = 1.1 + size_atom
    L_size_atom["max_wrt_grid"] = 0.075

    fields_tensor, betas_tensor, L_XYZ_grid = build_inputs_for_ML_parr(
        name_data_set,
        N_grid,
        L_bond_grid,
        L_size_atom,
        T_steps,
        N_molecule,
        num_processes,
    )

    LL_data_set_field = {}
    LL_data_set_field["name_data_set"] = name_data_set
    LL_data_set_field["N_grid"] = np.shape(fields_tensor)[2:]
    LL_data_set_field["L_bond_grid"] = L_bond_grid
    LL_data_set_field["L_XYZ_grid"] = L_XYZ_grid
    LL_data_set_field["L_size_atom"] = L_size_atom
    LL_data_set_field["T_steps"] = T_steps
    LL_data_set_field["N_molecule"] = N_molecule
    LL_data_set_field["fields_tensor"] = fields_tensor
    LL_data_set_field["betas_tensor"] = betas_tensor

    with open(name_for_save, "wb") as f:
        pickle.dump(LL_data_set_field, f)

    print(" Saved:", name_for_save)
