from typing import Union
from pathlib import Path

import torch.nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar

from .config import Config
from .neural_solver import NeuralSolver


class ConfigTrainer:
    def __init__(
        self, config: Config, output_dir: Union[None, Path, str], **trainer_kwargs
    ):
        self.config = config
        if output_dir is None:
            logger = False
            enable_checkpointing = False
        else:
            logger = TensorBoardLogger(
                str(output_dir),
                name="",
                default_hp_metric=False,
            )
            enable_checkpointing = True
        self.module = NeuralSolver(**config.get_module_params())
        self.trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=enable_checkpointing,
            callbacks=[LearningRateMonitor(), TQDMProgressBar(refresh_rate=100)],
            max_epochs=config.get_epochs(),
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            benchmark=True,
            detect_anomaly=True,
            **trainer_kwargs,
        )
        self.train_loader = config.get_train_loader()
        self.val_loader = config.get_test_loader()
        self.input_dim = self.train_loader.dataset.feature_dim

    def fit(self, model: torch.nn.Module) -> None:
        self.module.set_model(model)
        return self.trainer.fit(self.module, self.train_loader, self.val_loader)
