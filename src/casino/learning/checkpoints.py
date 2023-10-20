import logging
import pathlib
from typing import Any, Tuple, Union

try:
    import torch
except:
    logging.debug("torch not availble. Some functionality in checkpoints.py will break")


def save_torch_checkpoint(
    experiment_directory: Union[pathlib.Path, Any],
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    epoch: int,
    current_epoch: bool = False,
    latest: bool = True,
    best: bool = False,
):
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "checkpoints"
    experiment_directory.mkdir(parents=True, exist_ok=True)

    ckpt_names = []
    if current_epoch:
        ckpt_names.append(f"{epoch}")
    if latest:
        ckpt_names.append("latest")
    if best:
        ckpt_names.append("best")

    for ckpt_name in ckpt_names:
        torch.save(
            {
                "model_parameters": model.state_dict(),
                "optimizer_parameters": optimizer.state_dict(),
                "epoch": epoch,
            },
            experiment_directory / (ckpt_name + ".ckpt"),
        )


def load_torch_checkpoint(
    experiment_directory: Union[pathlib.Path, Any], ckpt_name: str = "best"
) -> Tuple["torch.nn.Module", "torch.optim.Optimizer", int]:
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "checkpoints"
    checkpoint_dict = torch.load(
        experiment_directory / (ckpt_name + ".ckpt"),
    )
    return (
        checkpoint_dict["model_parameters"],
        checkpoint_dict["optimizer_parameters"],
        checkpoint_dict["epoch"],
    )
