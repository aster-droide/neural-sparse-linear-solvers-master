#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
import pytorch_lightning as pl

from .config import Config
from .config_trainer import ConfigTrainer
from .neural_solver import NeuralSolver
from .single_inference import SingleInference


class CLI:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Command line interface for Neural Sparse Linear Solvers",
            usage=(
                "python3 -m nsls <command> [<args>]\n"
                "\n"
                "train       Train the model\n"
                "eval        Evaluate the model\n"
                "export      Export a trained model\n"
            ),
        )
        parser.add_argument(
            "command",
            type=str,
            help="Sub-command to run",
            choices=(
                "train",
                "eval",
                "export",
            ),
        )

        args = parser.parse_args(sys.argv[1:2])
        command = args.command.replace("-", "_")
        if not hasattr(self, command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)
        getattr(self, command)()

    @staticmethod
    def train() -> None:
        warnings.filterwarnings(
            "ignore",
            ".*Trying to infer the `batch_size` from an ambiguous collection.*",
        )
        parser = argparse.ArgumentParser(
            description="Train the model",
            usage="python3 -m nsls train config-path [--output-dir OUTPUT-DIR]",
        )
        parser.add_argument(
            "config_path",
            metavar="config-path",
            help="Path to configuration file",
        )
        parser.add_argument("--output-dir", help="Output directory", default="./runs")
        args = parser.parse_args(sys.argv[2:])

        config = Config(args.config_path)

        config_trainer = ConfigTrainer(
            config,
            Path(args.output_dir).expanduser(),
            gpus=1 if torch.cuda.is_available() else 0,
        )
        model = config.get_model(config_trainer.input_dim)
        config.save(config_trainer.trainer.logger.log_dir)
        config_trainer.fit(model)

    @staticmethod
    def eval() -> None:
        parser = argparse.ArgumentParser(
            description="Evaluate the model",
            usage="python3 -m nsls eval config-path --checkpoint CHECKPOINT",
        )
        parser.add_argument(
            "config_path",
            metavar="config-path",
            help="Path to configuration file",
        )
        parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
        args = parser.parse_args(sys.argv[2:])

        config = Config(args.config_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        test_loader = config.get_test_loader()
        model = config.get_model(test_loader.dataset.feature_dim)
        module = NeuralSolver(**checkpoint["hyper_parameters"])
        module.set_model(model)
        module.load_state_dict(checkpoint["state_dict"])
        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
        )
        results = trainer.test(module, dataloaders=test_loader)
        print(results)

    @staticmethod
    def export() -> None:
        parser = argparse.ArgumentParser(
            description="Export a trained model",
            usage="python3 -m nsls export config-path --checkpoint CHECKPOINT",
        )
        parser.add_argument(
            "config_path",
            metavar="config-path",
            help="Path to configuration file",
        )
        parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
        parser.add_argument(
            "--output-path", help="Output directory", default="model.pt"
        )
        parser.add_argument("--gpu", help="Export model for GPU", action="store_true")
        args = parser.parse_args(sys.argv[2:])

        config = Config(args.config_path)
        device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        test_dataset = config.get_test_dataset()

        model = config.get_model(test_dataset.feature_dim)
        processors = config.get_preprocessors()
        module = SingleInference(model, processors)
        module.load_state_dict(checkpoint["state_dict"])
        module = module.to(device).eval().requires_grad_(False)
        test_sample = test_dataset[0]
        test_inputs = (
            test_sample.b.to(device),
            test_sample.edge_index.to(device),
            test_sample.edge_attr.to(device),
        )
        traced_module = torch.jit.trace(
            module,
            test_inputs,
        )
        traced_module = torch.jit.freeze(traced_module)
        traced_module.save(args.output_path)


if __name__ == "__main__":
    CLI()
