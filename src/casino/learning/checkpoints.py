import logging
import pathlib
from typing import Any, Tuple, Union, Dict

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
    prefix: str = "",
    extras: Dict[str, Any] = {},
):
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "checkpoints"
    experiment_directory.mkdir(parents=True, exist_ok=True)

    ckpt_names = []
    if current_epoch:
        ckpt_names.append(f"{prefix}{epoch}")
    if latest:
        ckpt_names.append(f"{prefix}latest")
    if best:
        ckpt_names.append(f"{prefix}best")

    for ckpt_name in ckpt_names:
        torch.save(
            {
                "model_parameters": model.state_dict(),
                "optimizer_parameters": optimizer.state_dict(),
                "epoch": epoch,
                "extras": {**extras},
            },
            experiment_directory / (ckpt_name + ".ckpt"),
        )


def load_torch_checkpoint(
    experiment_directory: Union[pathlib.Path, Any],
    ckpt_name: str = "best",
    info: bool = False,
    **kwargs: Any,
) -> Tuple["torch.nn.Module", "torch.optim.Optimizer", int]:
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "checkpoints"
    ckpt_path = experiment_directory / (ckpt_name + ".ckpt")
    if info:
        logging.info(f"Loading checkpoint from {ckpt_path}")

    checkpoint_dict = torch.load(experiment_directory / (ckpt_name + ".ckpt"), **kwargs)

    return_values = (
        checkpoint_dict["model_parameters"],
        checkpoint_dict["optimizer_parameters"],
        checkpoint_dict["epoch"],
    )

    if "extras" not in checkpoint_dict:
        return return_values

    return return_values + (checkpoint_dict["extras"],)
