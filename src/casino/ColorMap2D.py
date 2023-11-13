# Credits go to: https://stackoverflow.com/questions/15207255/is-there-any-way-to-use-bivariate-colormaps-in-matplotlib/68981516#68981516

from typing import Tuple
from casino import DATA_DIR

CMAP_DIR = DATA_DIR / "ColorMaps2D"

try:
    import numpy as np
    import matplotlib.pyplot as plt
except:
    import logging

    logging.debug("numpy not availble. All functionality in ColoMap2D.py will break")

_AVAILABLE_COLOR_MAPS = [cmap_file.stem for cmap_file in CMAP_DIR.glob("*.png")]


class ColorMap2D:
    def __init__(
        self,
        color_map: str = "bremm",
        transpose=False,
        reverse_x=False,
        reverse_y=False,
        xclip=None,
        yclip=None,
        x_limits: Tuple[float] = (0.0, 1.0),
        y_limits: Tuple[float] = (0.0, 1.0),
    ):
        """
        Maps two 2D array to an RGB color space based on a given reference image.
        Args:
            filename (str): reference image to read the x-y colors from
            rotate (bool): if True, transpose the reference image (swap x and y axes)
            reverse_x (bool): if True, reverse the x scale on the reference
            reverse_y (bool): if True, reverse the y scale on the reference
            xclip (tuple): clip the image to this portion on the x scale; (0,1) is the whole image
            yclip  (tuple): clip the image to this portion on the y scale; (0,1) is the whole image
        """
        assert (
            color_map in _AVAILABLE_COLOR_MAPS
        ), f"{color_map} not in {_AVAILABLE_COLOR_MAPS}"
        self._colormap_file = CMAP_DIR / f"{color_map}.png"
        self._img = plt.imread(self._colormap_file)
        if transpose:
            self._img = self._img.transpose()
        if reverse_x:
            self._img = self._img[::-1, :, :]
        if reverse_y:
            self._img = self._img[:, ::-1, :]
        if xclip is not None:
            imin, imax = map(lambda x: int(self._img.shape[0] * x), xclip)
            self._img = self._img[imin:imax, :, :]
        if yclip is not None:
            imin, imax = map(lambda x: int(self._img.shape[1] * x), yclip)
            self._img = self._img[:, imin:imax, :]
        if issubclass(self._img.dtype.type, np.integer):
            self._img = self._img / 255.0

        self._width = len(self._img)
        self._height = len(self._img[0])

        self._range_x = x_limits
        self._range_y = y_limits

    @staticmethod
    def _scale_to_range(u: "np.ndarray", u_min: float, u_max: float) -> "np.ndarray":
        return (u - u_min) / (u_max - u_min)

    def _map_to_x(self, val: "np.ndarray") -> "np.ndarray":
        xmin, xmax = self._range_x
        val = self._scale_to_range(val, xmin, xmax)
        rescaled = val * (self._width - 1)
        return rescaled.astype(int)

    def _map_to_y(self, val: "np.ndarray") -> "np.ndarray":
        ymin, ymax = self._range_y
        val = self._scale_to_range(val, ymin, ymax)
        rescaled = val * (self._height - 1)
        return rescaled.astype(int)

    def __call__(self, val_x, val_y, update_range: bool = False):
        """
        Take val_x and val_y, and associate the RGB values
        from the reference picture to each item. val_x and val_y
        must have the same shape.
        """
        if val_x.shape != val_y.shape:
            raise ValueError(
                f"x and y array must have the same shape, but have {val_x.shape} and {val_y.shape}."
            )
        if update_range:
            self._range_x = (np.amin(val_x), np.amax(val_x))
            self._range_y = (np.amin(val_y), np.amax(val_y))
        x_indices = self._map_to_x(val_x)
        y_indices = self._map_to_y(val_y)
        i_xy = np.stack((x_indices, y_indices), axis=-1)
        rgb = np.zeros((*val_x.shape, 3))
        for indices in np.ndindex(val_x.shape):
            img_indices = tuple(i_xy[indices])
            rgb[indices] = self._img[img_indices]
        return rgb

    def generate_cbar(self, nx=100, ny=100):
        "generate an image that can be used as a 2D colorbar"
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        return self.__call__(*np.meshgrid(x, y), update_range=True)
