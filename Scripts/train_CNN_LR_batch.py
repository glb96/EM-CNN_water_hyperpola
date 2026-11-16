"""EM-CNN architecture and staged training (Keras with the PyTorch backend)."""
import os
from datetime import datetime
from pathlib import Path

os.environ['KERAS_BACKEND'] = 'torch'
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import keras
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch_dtype = torch.float32

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

DEFAULT_SCHEDULE = [
    {"epochs": 20, "batch_size": 256, "lr": 1e-3},
    {"epochs": 15, "batch_size": 64, "lr": 5e-4},
    {"epochs": 30, "batch_size": 128, "lr": 2.5e-4},
    {"epochs": 30, "batch_size": 64, "lr": 1e-4},
    {"epochs": 75, "batch_size": 128, "lr": 1e-4},
]


def train_model(*, train_path, test_path, run_dir, schedule=None,
                shuffle_train=False, seed=42, clipnorm=1.0,
                reduce_lr_factor=0.5, reduce_lr_patience=5, min_lr=1e-6,
                early_stopping_patience=15, save_each_epoch=True):
    """Train with the original architecture, normalization and phase settings.

    The test split monitors optimization; final assessment uses valid elsewhere.
    Each phase retains the original fit/early-stopping behavior. Epoch numbering
    is cumulative so per-epoch files and TensorBoard steps do not overlap.
    """
    schedule = DEFAULT_SCHEDULE if schedule is None else schedule
    if not schedule or any(s['epochs'] < 1 or s['batch_size'] < 1 or s['lr'] <= 0
                           for s in schedule):
        raise ValueError("Schedule must contain positive epochs, batch_size and lr")
    keras.utils.set_random_seed(seed)
    run_dir = Path(run_dir).resolve()
    (run_dir / 'models').mkdir(parents=True, exist_ok=True)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
    train_dataset = torch.load(train_path, map_location='cpu', weights_only=False)
    test_dataset = torch.load(test_path, map_location='cpu', weights_only=False)
    x0, y0 = train_dataset[0]
    N_grid, N_beta = tuple(x0.shape[:3]), int(y0.shape[0])
    if len(x0.shape) != 4 or x0.shape[-1] != 3 or N_beta != 18:
        raise ValueError("Expected channels-last maps and 18 beta components")
    if test_dataset[0][0].shape != x0.shape:
        raise ValueError("Training and test grid shapes differ")
    if min(N_grid) < 8:
        raise ValueError("Each cropped grid dimension must be >= 8 for three pooling layers")
    print("N_grid=", N_grid, "N_beta=", N_beta)
    model = get_model_01(N_grid=N_grid, N_beta=N_beta)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=schedule[0]['lr'], clipnorm=clipnorm),
        loss="mse", metrics=["mae", R2Score()],
    )
    tensorboard = TensorboardCallback(
        log_dir=str(run_dir / 'logs' / datetime.now().strftime('tb_%Y%m%d_%H%M%S')))
    # One checkpoint object across phases retains the best validation loss.
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir / 'models' / 'best-model.keras'), monitor='val_loss',
        mode='min', save_best_only=True, verbose=1)
    callbacks = [tensorboard, checkpoint]
    if save_each_epoch:
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / 'models' / '{epoch:03d}.keras'),
            save_freq='epoch', verbose=0))
    callbacks.extend([
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', mode='min', factor=reduce_lr_factor,
            patience=reduce_lr_patience, min_lr=min_lr, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=early_stopping_patience,
            restore_best_weights=True, verbose=1),
    ])
    initial_epoch = 0
    try:
        for phase in schedule:
            train_loader = DataLoader(
                dataset=train_dataset, batch_size=phase['batch_size'],
                shuffle=shuffle_train, num_workers=0, pin_memory=torch.cuda.is_available(),
                drop_last=False)
            test_loader = DataLoader(
                dataset=test_dataset, batch_size=phase['batch_size'],
                shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(),
                drop_last=False)
            model.optimizer.learning_rate.assign(phase['lr'])
            history = model.fit(
                train_loader, initial_epoch=initial_epoch,
                epochs=initial_epoch + phase['epochs'], validation_data=test_loader,
                callbacks=callbacks, shuffle=False)
            initial_epoch += len(history.epoch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model.save(run_dir / 'final_model.keras')
    finally:
        tensorboard.writer.close()
    return run_dir / 'models' / 'best-model.keras'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-path', default='../train_CNN_small.p')
    parser.add_argument('--test-path', default='../test_CNN_small.p')
    parser.add_argument('--run-dir', default='.')
    args = parser.parse_args()
    train_model(**vars(args))
