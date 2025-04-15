"""Utility functions for training, evaluation and visualization."""

from torch import nn

def tqdm_bar(mode: str, pbar, target: float = 0.0, cur_epoch: int = 0, epochs: int = 0) -> None:
    """
    Update the tqdm progress bar with custom format.

    Args:
        mode (str): Current mode ('Train', 'Val', 'Test').
        pbar: tqdm progress bar instance.
        target (float): Current loss value.
        cur_epoch (int): Current epoch.
        epochs (int): Total number of epochs.
    """
    if mode == 'Test':
        pbar.set_description(f"({mode})", refresh=False)
    else:
        pbar.set_description(f"({mode}) Epoch {cur_epoch}/{epochs}", refresh=False)
        pbar.set_postfix(loss=float(target), refresh=False)
    pbar.refresh()

def print_model_params(model: nn.Module) -> None:
    """
    Print the model architecture and total number of parameters.

    Args:
        model (nn.Module): PyTorch model
    """
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print("# Parameters:", total_params)
