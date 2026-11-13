import importlib
import numpy as np
import sys
import os
import pickle
import time
import psutil
import gc

import torch
from torch.utils.data import TensorDataset

import argparse

np.set_printoptions(precision=6, suppress=True)  # Optional: improves readability
np.seterr(all="ignore")  # Ignore floating-point warnings (optional)

# Change default dtype for numpy arrays
np_dtype = np.float32  # Define global dtype
torch_dtype = torch.float32



############# SPC/E bulk ##################
beta_mean_ref = np.array([0.0, 0.0, -3.489448070526123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0067856311798096, 0.0, -3.45514178276062, 0.0, 0.0, 3.381779670715332, 0.0, 5.3631463050842285])
beta_std_ref = np.array([2.9813599586486816, 2.658426523208618, 2.1298675537109375, 3.052894115447998, 1.541837453842163, 2.16447377204895, 2.8199944496154785, 3.2426583766937256, 1.5952163934707642, 10.646726608276367,
      3.533592939376831, 3.1656177043914795, 2.1693038940429688, 1.5565845966339111, 2.207494020462036, 3.3727798461914062, 3.0721993446350098, 5.716089725494385])

normalization_stats = {
      "field_means": [],
      "field_stds": [],
      "beta_means": beta_mean_ref,
      "beta_stds" : beta_std_ref}

############# SPC/E bulk ##################

###########################################################################
###########################   Utilities   #################################
###########################################################################

def reshape_memmap_tensor(tensor, reshaping):
    """
    Reshape/slice a tensor. If it is a memmap, rewrite the underlying file to only keep selected indices.
    """
    if isinstance(tensor, np.memmap):
        old_filename = tensor.filename
        shape_tensor = tensor.shape
        new_shape = (len(reshaping),) + shape_tensor[1:]
        tmp_filename = "temp_filename.dat"
        new_tensor = np.memmap(tmp_filename, dtype=np_dtype, mode="w+", shape=new_shape)
        new_tensor[:] = tensor[reshaping]
        del tensor
        gc.collect()
        if os.path.exists(old_filename):
            os.remove(old_filename)
        os.rename(tmp_filename, old_filename)
        new_tensor = np.memmap(old_filename, dtype=np_dtype, mode="r+", shape=new_shape)
        return new_tensor
    else:
        return tensor[reshaping]


def masked_mean_std(tensor):
    """
    Ignore zeros (grid is sparse). Compute mean/std only on non-zero values.
    """
    mask = np.abs(tensor) > 1e-9
    if not np.any(mask):
        # Avoid NaNs if everything is zero
        return 0.0, 1.0
    mean = np.mean(tensor[mask])
    std = np.std(tensor[mask])
    if std < 1e-12:
        std = 1.0
    return mean, std


def print_normalization_stats(field_means, field_stds, beta_means, beta_stds):
    print("################ NORMALIZATION STATS ################")
    print("Field means (per component):", field_means)
    print("Field stds  (per component):", field_stds)
    print("Beta means  (per component):", beta_means)
    print("Beta stds   (per component):", beta_stds)
    print("#####################################################")


###########################################################################
###########################   I/O data   ##################################
###########################################################################

def read_data(name_data_for_training, memory_saving=True):
    """
    Load the dataset pickle and return (L_XYZ_grid, N_grid, fields_tensor, betas_tensor).

    If memory_saving=True, fields_tensor is stored as a memmap on disk (fields_tensor.dat).
    """
    with open(name_data_for_training, "rb") as filetoload:
        LL_data_set_field = pickle.load(filetoload)

    N_grid = LL_data_set_field["N_grid"]
    L_XYZ_grid = LL_data_set_field["L_XYZ_grid"]

    fields_tensor = LL_data_set_field["fields_tensor"]
    betas_tensor = LL_data_set_field["betas_tensor"]

    # Convert torch -> numpy if needed
    if hasattr(fields_tensor, "numpy"):
        fields_tensor = fields_tensor.numpy()
    if hasattr(betas_tensor, "numpy"):
        betas_tensor = betas_tensor.numpy()

    # Ensure numpy dtype
    if memory_saving:
        fields_tensor_t = np.memmap("fields_tensor.dat", dtype=np_dtype, mode="w+", shape=fields_tensor.shape)
        fields_tensor_t[:] = fields_tensor[:]
        fields_tensor = fields_tensor_t
    else:
        fields_tensor = fields_tensor.astype(dtype=np_dtype, copy=True)

    betas_tensor = betas_tensor.astype(dtype=np_dtype, copy=True)

    # Free the original dict asap
    LL_data_set_field.clear()
    del LL_data_set_field
    gc.collect()

    return (L_XYZ_grid, N_grid, fields_tensor, betas_tensor)


###########################################################################
#########################   Processing   ##################################
###########################################################################

def resize_data_nbr_sample(N_config, ratio_data_to_use, fields_tensor, betas_tensor):
    """
    Optionally keep only the first ratio_data_to_use fraction of data.
    """
    if ratio_data_to_use != 1:
        print("Nbr config before reshape:", N_config)
        N_config_to_use = int(N_config * ratio_data_to_use)
        fields_tensor = reshape_memmap_tensor(fields_tensor, [k for k in range(N_config_to_use)])
        betas_tensor = betas_tensor[:N_config_to_use]
        N_config = len(betas_tensor)
        print("Nbr config after reshape:", N_config)
    return (N_config, fields_tensor, betas_tensor)


def modify_e_on_the_grid(L_XYZ_grid, N_grid, fields_tensor, use_r2_1=False, use_log_mean=False):
    """
    Optional grid-wise transformation of the electrostatic field.
    """
    X, Y, Z = L_XYZ_grid
    if use_r2_1:
        print("Grid modification using 1/(1+r^2)")
        L_r2 = X**2 + Y**2 + Z**2
        print("Maximal r**2 value on the grid", np.max(L_r2))
        fields_tensor[:] /= (1.0 + L_r2)
    elif use_log_mean:
        print("Grid modification using log type")
        mean_field, _ = masked_mean_std(np.abs(fields_tensor))
        print("Mean value of |field|", mean_field)
        fields_tensor[:] = np.sign(fields_tensor) * np.log10(1.0 + np.abs(fields_tensor) / (mean_field + 1e-12))
    else:
        print("No grid modification")
    return fields_tensor


def normalizing_field_beta(fields_tensor, betas_tensor, normalization_stats=None, print_stats=True):
    """
    Normalize fields and betas.

    If normalization_stats is None:
        - compute masked mean/std on current dataset
        - print them (if print_stats=True)
        - return (fields_tensor, betas_tensor, stats_dict)
    If normalization_stats is provided:
        - use provided stats (no recomputation)
        - return (fields_tensor, betas_tensor, normalization_stats)

    normalization_stats format:
    {
        "field_means": [mEx, mEy, mEz],
        "field_stds":  [sEx, sEy, sEz],
        "beta_means":  [.. length B ..],
        "beta_stds":   [.. length B ..],
    }
    """
    N, C, *_ = fields_tensor.shape
    _, B = betas_tensor.shape

    if normalization_stats is None:
        field_means = []
        field_stds = []
        beta_means = []
        beta_stds = []
    else:
        stats = normalization_stats
        field_means = stats["field_means"]
        field_stds = stats["field_stds"]
        beta_means = stats["beta_means"]
        beta_stds = stats["beta_stds"]

    if len(field_means) < 2:
        for i in range(C):
            m, s = masked_mean_std(fields_tensor[:, i, :, :, :])
            field_means.append(float(m))
            field_stds.append(float(s))
    if len(beta_means) < 2:
        beta_means = []
        beta_stds = []
        for i in range(B):
            m, s = masked_mean_std(betas_tensor[:, i])
            beta_means.append(float(m))
            beta_stds.append(float(s))
   
    stats = {
            "field_means": field_means,
            "field_stds": field_stds,
            "beta_means": beta_means,
            "beta_stds": beta_stds,
            }

    if print_stats:
        print_normalization_stats(field_means, field_stds, beta_means, beta_stds)

    # Normalize field: only non-zero voxels
    for i in range(C):
        mask = fields_tensor[:, i, :, :, :] != 0
        fields_tensor[:, i, :, :, :][mask] -= field_means[i]
        fields_tensor[:, i, :, :, :][mask] /= (field_stds[i])

    # Normalize betas: all entries
    for i in range(B):
        betas_tensor[:, i] = (betas_tensor[:, i] - beta_means[i]) / (beta_stds[i])

    return fields_tensor, betas_tensor, stats


###########################################################################
#######################   Data augmentation   #############################
###########################################################################

def flip_field_beta(my_field, my_beta, along_x=False, along_y=False):
    """
    Flip field + beta according to symmetry.

    my_field shape: (3, Nx, Ny, Nz)
    my_beta  shape: (18,)  (already flattened)
    """
    flipped_field = np.copy(my_field)
    flipped_beta = np.copy(my_beta)

    if along_x and not along_y:
        flipped_field[0] = -1 * flipped_field[0]         # invert E_x
        flipped_field = flipped_field[:, ::-1, :, :]      # flip x-axis

        # odd apparition of x axis should flip sign for beta:
        indices_to_flip = [0, 3, 4, 5, 7, 8, 13, 14]
        flipped_beta[indices_to_flip] *= -1

    elif (not along_x) and along_y:
        flipped_field[1] = -1 * flipped_field[1]         # invert E_y
        flipped_field = flipped_field[:, :, ::-1, :]      # flip y-axis

        # odd apparition of y axis should flip sign for beta:
        indices_to_flip = [1, 4, 6, 8, 9, 11, 13, 16]
        flipped_beta[indices_to_flip] *= -1

    elif along_x and along_y:
        flipped_field[0] = -1 * flipped_field[0]
        flipped_field[1] = -1 * flipped_field[1]
        flipped_field = flipped_field[:, ::-1, ::-1, :]   # flip x and y

        # odd apparition of (x+y) axis should flip sign for beta:
        indices_to_flip = [0, 1, 3, 5, 6, 7, 9, 11, 14, 16]
        flipped_beta[indices_to_flip] *= -1

    else:
        raise Exception("CODE ERROR: this should not happen...")

    return flipped_field, flipped_beta


def sym_duplication(fields_tensor, betas_tensor, along_x=True, along_y=True):
    """
    Data augmentation using symmetry.
    Keeps same behavior: duplicates dataset by factor 2 or 4 depending on flips.

    If fields_tensor is memmap, output is a memmap ("new_field_tensor.dat").
    """
    if (not along_x) and (not along_y):
        return fields_tensor, betas_tensor

    oldfilename = fields_tensor.filename if isinstance(fields_tensor, np.memmap) else None

    N_config = len(betas_tensor)
    N_shape = fields_tensor.shape  # (N,3,Nx,Ny,Nz)
    N_shape_beta = betas_tensor.shape  # (N,18)

    if along_x and along_y:
        new_N_config = 4 * N_config
    else:
        new_N_config = 2 * N_config

    new_field_tensor = np.memmap(
        "new_field_tensor.dat",
        dtype=np_dtype,
        mode="w+",
        shape=(new_N_config, 3, N_shape[2], N_shape[3], N_shape[4]),
    )
    new_field_tensor[:] = 0
    new_betas_tensor = np.zeros((new_N_config, N_shape_beta[1]), dtype=np_dtype)

    trotter = 0
    for n in range(N_config):
        my_field = np.array(fields_tensor[n], copy=True)
        my_beta = np.array(betas_tensor[n], copy=True)

        new_field_tensor[trotter] = my_field
        new_betas_tensor[trotter] = my_beta
        trotter += 1

        if along_x:
            f, b = flip_field_beta(my_field, my_beta, along_x=True, along_y=False)
            new_field_tensor[trotter] = f
            new_betas_tensor[trotter] = b
            trotter += 1

        if along_y:
            f, b = flip_field_beta(my_field, my_beta, along_x=False, along_y=True)
            new_field_tensor[trotter] = f
            new_betas_tensor[trotter] = b
            trotter += 1

        if along_x and along_y:
            f, b = flip_field_beta(my_field, my_beta, along_x=True, along_y=True)
            new_field_tensor[trotter] = f
            new_betas_tensor[trotter] = b
            trotter += 1

    # clean old memmap file if any, and replace it with the new one
    del fields_tensor
    gc.collect()

    if oldfilename is not None:
        # Replace old file with the new file name
        if os.path.exists(oldfilename):
            os.remove(oldfilename)
        os.rename("new_field_tensor.dat", oldfilename)
        new_field_tensor = np.memmap(oldfilename, dtype=np_dtype, mode="r+", shape=new_field_tensor.shape)
    else:
        # Keep as "new_field_tensor.dat"
        new_field_tensor.flush()

    return new_field_tensor, new_betas_tensor


###########################################################################
#######################   Main preparation   ##############################
###########################################################################

def prepare_data_for_GNN_ready_to_eat(
    name_data_for_training,
    name_data_set,
    ratio_data_to_use=1.0,
    memory_saving=True,
    use_r2_1=False,
    use_log_mean=False,
    do_sym_augmentation=True,
    aug_along_x=True,
    aug_along_y=True,
    normalization_stats=None,
    print_stats=True,
):
    """
    Prepare ONE dataset (already known to be train OR test OR val externally).
    This function does NOT split. It outputs a single TensorDataset saved with torch.save.

    Steps kept:
      - read_data (with memmap option)
      - optional resize via ratio_data_to_use
      - optional grid modification
      - optional symmetry augmentation
      - normalization (internal stats or provided external stats)
      - transpose to channels-last (as you were doing)
      - save ONE dataset only

    Also prints normalization stats if they were computed (so you can reuse later).
    """
    L_XYZ_grid, N_grid, fields_tensor, betas_tensor = read_data(name_data_for_training, memory_saving=memory_saving)
    N_config = len(betas_tensor)
    print("Initial N_config:", N_config)

    # Reduce sample count if needed
    N_config, fields_tensor, betas_tensor = resize_data_nbr_sample(N_config, ratio_data_to_use, fields_tensor, betas_tensor)

    # Grid modification
    fields_tensor = modify_e_on_the_grid(L_XYZ_grid, N_grid, fields_tensor, use_r2_1=use_r2_1, use_log_mean=use_log_mean)

    # Symmetry augmentation (optional)
    if do_sym_augmentation:
        print("Before augmentation:", len(betas_tensor))
        fields_tensor, betas_tensor = sym_duplication(fields_tensor, betas_tensor, along_x=aug_along_x, along_y=aug_along_y)
        print("After augmentation:", len(betas_tensor))
    else:
        print("No augmentation requested.")

    # Normalization: internal or external stats
    fields_tensor, betas_tensor, used_stats = normalizing_field_beta(
        fields_tensor,
        betas_tensor,
        normalization_stats=normalization_stats,
        print_stats=print_stats,
    )

    # Channels last (kept as in your code)
    fields_tensor = np.transpose(fields_tensor, (0, 2, 3, 4, 1))  # (N, Nx, Ny, Nz, 3)
    print("Final fields shape:", fields_tensor.shape)
    print("Final betas shape :", betas_tensor.shape)

    # Convert to torch dataset (single dataset)
    fields_torch = torch.from_numpy(np.array(fields_tensor, copy=False)).to(dtype=torch_dtype)
    betas_torch = torch.from_numpy(np.array(betas_tensor, copy=False)).to(dtype=torch_dtype)
    dataset = TensorDataset(fields_torch, betas_torch)

    # Save
    torch.save(dataset, name_data_set)
    print("Saved dataset:", name_data_set)

    # Also optionally save the normalization stats next to it (handy reuse)
    # (You can comment this out if you don't want extra files)
    stats_path = os.path.splitext(name_data_set)[0] + "_normalization_stats.p"
    with open(stats_path, "wb") as f:
        pickle.dump(used_stats, f)
    print("Saved normalization stats:", stats_path)

    # Cleanup memmap file if it exists
    if isinstance(fields_tensor, np.memmap):
        filename = fields_tensor.filename
        del fields_tensor
        gc.collect()
        if os.path.exists(filename):
            os.remove(filename)

    # Also cleanup default memmap name if still around
    if os.path.exists("fields_tensor.dat"):
        try:
            os.remove("fields_tensor.dat")
        except Exception:
            pass

    return used_stats


###########################################################################
###############################   RUN   ###################################
###########################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare electrostatic maps dataset for CNN/GNN")
    parser.add_argument("name_data_for_training", type=str, help="Input pickle (.p) electrostatic maps")
    parser.add_argument("name_data_set", type=str, help="Output dataset file (.p or .pt)")

    args = parser.parse_args()

    name_data_for_training = args.name_data_for_training
    name_data_set = args.name_data_set

    ratio_data_to_use = 1.0

    # Augmentation options
    do_sym_augmentation = True
    aug_along_x = True
    aug_along_y = True

    # Grid modification options
    use_r2_1 = False
    use_log_mean = False

    # Normalization
    # normalization_stats = None
    normalization_stats = normalization_stats
    print_stats = True

    prepare_data_for_GNN_ready_to_eat(
        name_data_for_training=name_data_for_training,
        name_data_set=name_data_set,
        ratio_data_to_use=ratio_data_to_use,
        memory_saving=True,
        use_r2_1=use_r2_1,
        use_log_mean=use_log_mean,
        do_sym_augmentation=do_sym_augmentation,
        aug_along_x=aug_along_x,
        aug_along_y=aug_along_y,
        normalization_stats=normalization_stats,
        print_stats=print_stats,
    )
