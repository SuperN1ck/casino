# TODO Make this work
# import logging
# import importlib
# def guarded_import(module: str):
#     try:
#         i = importlib.import_module("matplotlib.text")
#         return i # TODO Does this work like this?
#     except:
#         logging.debug(f"Can not import {module}, some functionality might be broken.")

import pathlib

BASE_DIR = pathlib.Path(__file__).parents[2]
DATA_DIR = BASE_DIR / "data"


from . import (
    cache,
    compress,
    dataclasses,
    experiments,
    geometry,
    hardware,
    latents,
    learning,
    masks,
    notebooks,
    pointcloud,
    random,
    special_dicts,
    visualization,
    math,
    images,
    user,
    o3d,
    torch_utils,
)
from .ColorMap2D import ColorMap2D

__all__ = [
    "cache",
    "compress",
    "dataclasses",
    "experiments",
    "learning",
    "notebooks",
    "pointcloud",
    "random",
    "special_dicts",
    "visualization",
    "ColorMap2D",
    "latents",
    "geometry",
    "images",
    "math",
    "user",
    "masks",
    "hardware",
    "o3d",
    "torch_utils",
]
