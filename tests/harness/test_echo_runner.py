import jax
import pytest
import torch
import os
import yaml
from torch.utils.data import DataLoader, Dataset

from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.harness.echo_trainer import EchoTrainer
from src.echo.harness.echo_runner import EchoRunner


class DummyDataset(Dataset):
    def __init__(self, size, seq_len, d_sensory, d_state):
        self.size = size
        self.s_true = torch.randn(size, seq_len, d_sensory)
        self.x_init = torch.randn(size, d_state)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"s_true": self.s_true[idx], "x_init": self.x_init[idx]}


def test_echo_runner_orchestration(tmp_path):
    """
    ARRANGE
    """
    # Create temporary config
    config_path = tmp_path / "test_config.yaml"
    config_data = {"optimization": {"learning_rate": 1e-3, "max_grad_norm": 1.0, "max_epochs": 2}}
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    d_internal = 2
    d_sensory = 2
    d_active = 2
    d_external = 2
    d_state = d_internal + d_sensory + d_active + d_external
    seq_len = 5

    key = jax.random.PRNGKey(42)
    k_model, k_run = jax.random.split(key)

    model = MarkovBlanketObserver(
        d_internal=d_internal,
        d_sensory=d_sensory,
        d_active=d_active,
        d_external=d_external,
        ebm_hidden_size=8,
        ebm_depth=1,
        n_steps=5,
        temperature=1.0,
        key=k_model,
    )

    trainer = EchoTrainer(model=model, learning_rate=1e-3)
    runner = EchoRunner(str(config_path))
    runner.setup(trainer)

    train_dataset = DummyDataset(16, seq_len, d_sensory, d_state)
    val_dataset = DummyDataset(16, seq_len, d_sensory, d_state)

    train_loader = DataLoader(train_dataset, batch_size=8)
    val_loader = DataLoader(val_dataset, batch_size=8)

    """
    ACT
    """
    # We will test train_epoch and validate
    model, train_loss = runner.train_epoch(model, train_loader, k_run, dt=0.01)
    val_loss, hessian_trace = runner.validate(model, val_loader, k_run, dt=0.01)

    """
    ASSERT
    """
    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)
    assert isinstance(hessian_trace, float)

    # Run loop
    runner.run(model, train_loader, val_loader, k_run, dt=0.01)
