import importlib
import numpy as np
import sys
import os 
import pickle
import time
import psutil

import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import fidle
import keras

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from keras.layers import BatchNormalization
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float32


###########################################################################
##############          Others      #######################################
###########################################################################

class R2Score(keras.metrics.Metric):
    def __init__(self, name="r2", eps=1e-12, **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
        self.sse = self.add_weight(name="sse", initializer="zeros")
        self.sst = self.add_weight(name="sst", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convertir et aligner device + dtype
        y_pred = torch.as_tensor(y_pred)
        y_true = torch.as_tensor(y_true, device=y_pred.device, dtype=y_pred.dtype)

        # moyenne sur le batch
        y_mean = torch.mean(y_true)

        sse = torch.sum((y_true - y_pred) ** 2)
        sst = torch.sum((y_true - y_mean) ** 2)

        # self.sse/self.sst sont des variables Keras -> parfois CPU
        # On ramène les scalaires sur le device/dtype des variables avant assign_add.
        sse = sse.to(self.sse.value.device, dtype=self.sse.value.dtype)
        sst = sst.to(self.sst.value.device, dtype=self.sst.value.dtype)

        self.sse.assign_add(sse)
        self.sst.assign_add(sst)

    def result(self):
        eps = torch.tensor(self.eps, device=self.sse.value.device, dtype=self.sse.value.dtype)
        return 1.0 - self.sse / (self.sst + eps)

    def reset_state(self):
        self.sse.assign(0.0)
        self.sst.assign(0.0)

class TensorboardCallback(keras.callbacks.Callback):
    def __init__(self, log_dir=None):
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # MAE
        if "mae" in logs and "val_mae" in logs:
            self.writer.add_scalars(
                "MAE",
                {"Train": logs["mae"], "Validation": logs["val_mae"]},
                epoch
            )

        # Loss
        if "loss" in logs and "val_loss" in logs:
            self.writer.add_scalars(
                "Loss",
                {"Train": logs["loss"], "Validation": logs["val_loss"]},
                epoch
            )

        # R2
        if "r2" in logs and "val_r2" in logs:
            self.writer.add_scalars(
                "R2",
                {"Train": logs["r2"], "Validation": logs["val_r2"]},
                epoch
            )

        self.writer.flush()

###########################################################################
##############          Others      #######################################
###########################################################################
###########################################################################
##############          CNN      ##########################################
###########################################################################

def get_model_01(N_grid, N_beta):

    lx, ly, lz = N_grid
    model = keras.models.Sequential()

    # Input layer
    model.add(keras.layers.Input(shape=(lx, ly, lz, 3), dtype=torch_dtype))  # Adding 3 channels (Ex, Ey, Ez)

    # 3D Convolutional Layers
    model.add(keras.layers.AveragePooling3D((2, 2, 2)))
    model.add(keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', dtype=torch_dtype))
    model.add(keras.layers.AveragePooling3D((2, 2, 2)))
    model.add(keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', dtype=torch_dtype))
    # model.add(keras.layers.Dropout(0.05))
    model.add(keras.layers.AveragePooling3D((2, 2, 2)))
    model.add(keras.layers.Conv3D(64, (2, 2, 2), activation='relu', padding='same', dtype=torch_dtype))
    
    # dense NN
    model.add(keras.layers.Flatten()) 
    model.add(keras.layers.Dense(256, activation='relu', dtype=torch_dtype))
    model.add(keras.layers.Dense(512, activation='relu', dtype=torch_dtype))
    model.add(keras.layers.Dense(1024, activation='relu', dtype=torch_dtype))
    # model.add(keras.layers.Dropout(0.05))
    model.add(keras.layers.Dense(1024, activation='relu', dtype=torch_dtype))
    # model.add(keras.layers.Dropout(0.05))
    # model.add(BatchNormalization())  # Add BatchNormalization for stability
    model.add(keras.layers.Dense(512, activation='relu', dtype=torch_dtype))

    #model.add( keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(N_beta, activation='linear', name='Output', dtype=torch_dtype))
    
    return model

###########################################################################
##############          CNN      ##########################################
###########################################################################

###########################################################################

name_train_set = "../train_CNN_small.p"
train_dataset = torch.load(name_train_set)

name_test_set = "../test_CNN_small.p"
test_dataset = torch.load(name_test_set)

def make_loaders(batch_size: int):
    num_workers = 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# Convert datasets to tensors (keeping everything in PyTorch)
x0, y0 = train_dataset[0]  # un seul exemple
                        
N_grid = x0.shape[0:3]
N_beta = y0.shape[0] if y0.ndim > 0 else 1

print("N_grid=", N_grid)
print("N_beta=", N_beta)


###########################################################################

run_dir = '.' 
os.makedirs(run_dir + '/models', exist_ok=True)
os.makedirs(run_dir + '/logs', exist_ok=True)


tenseorboard_callback = TensorboardCallback(
    log_dir=run_dir + "/logs/tb_" + fidle.Chrono.tag_now()
)

bestmodel_callback = keras.callbacks.ModelCheckpoint(
    filepath=run_dir + "/models/best-model.keras",
    monitor="val_loss",
    mode="min",               
    save_best_only=True,
    verbose=1
)

savemodel_callback = keras.callbacks.ModelCheckpoint(
    filepath=run_dir + "/models/{epoch:02d}.keras",
    save_freq="epoch",
    verbose=0
)

reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

earlystop_callback = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=15,
    restore_best_weights=True,   # <-- makes final model = best val_loss weights
    verbose=1
)

callbacks = [
    tenseorboard_callback,
    bestmodel_callback,
    savemodel_callback,
    reduce_lr_callback,
    earlystop_callback
]


###########################################################################

model = get_model_01(N_grid, N_beta)
model.summary()

###########################################################################

# ---- Train
epochs = 200
batch_size = 128
learning_rate = 1e-3
optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)

model.compile(optimizer=optimizer, loss="mse", metrics=["mae", R2Score()])

schedules = [
    {"epochs": 20, "batch_size": 256, "lr": 1e-3},
    {"epochs": 15, "batch_size": 64, "lr": 5e-4},
    {"epochs": 30, "batch_size": 128,  "lr": 2.5e-4},
    {"epochs": 30, "batch_size": 64,  "lr": 1e-4},
    {"epochs": 75, "batch_size": 128,  "lr": 1e-4},
]

for s in schedules:
    batch_size = s["batch_size"]
    epochs = s["epochs"]
    lr = s["lr"]
    train_loader, val_loader = make_loaders(batch_size)
    model.optimizer.learning_rate.assign(lr)
    model.fit(
        train_loader,
        epochs=epochs,
        validation_data=val_loader,
        callbacks=callbacks,
        shuffle=False
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

model.save("final_model.keras")

###########################################################################



