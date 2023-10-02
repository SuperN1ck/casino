import dataclasses
import logging
import pathlib
from typing import Any, Type, Union

try:
    import tyro
except:
    logging.debug("tyro not availble. Some functionality in dataclasses.py will break")


def save_cfg(
    cfg: Type[dataclasses.dataclass],
    experiment_directory: Union[pathlib.Path, str] = ".",
    experiment_id: str = "experiment_id",  # Should be an attribute of the class?
    file_name: str = "config.yaml",
):
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "runs"
    experiment_directory /= str(getattr(cfg, experiment_id))
    experiment_directory.mkdir(parents=True, exist_ok=True)
    with open(experiment_directory / file_name, "w") as file:
        file.write(tyro.to_yaml(cfg))
    return experiment_directory


# TODO NH?
def load_cfg(
    experiment_directory: Union[pathlib.Path, Any],
    name: str = "config.yaml",
    # TODO figure out a default class here or make it positional?
    cfg_class: Type[dataclasses.dataclass] = "",
) -> Type[dataclasses.dataclass]:
    with open(experiment_directory / name, "r") as file:
        cfg = tyro.from_yaml(cfg_class, file.read())
    return cfg
