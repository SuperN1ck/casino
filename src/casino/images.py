from typing import List, Union, Optional
import pathlib

try:
    import numpy as np
except:
    import logging

    logging.debug("numpy not availble. Most functionality in images.py will break.")

try:
    from PIL import Image
except:
    logging.debug("PIl not availble. Most functionality in images.py will break.")



def save_images_to_gif(
    images: List["np.ndarray"],
    output_path: Union[pathlib.Path, str],
    fps: Optional[float] = None,
    duration: Optional[float] = None,
):
    assert not (duration is None and fps is None)

    if duration is None:
        duration = len(images) / fps
    
    im_0 = Image.fromarray(images[0])
    im_0.save(
        pathlib.Path(output_path),
        format="GIF",
        append_images=[Image.fromarray(im) for im in images],
        save_all=True,
        duration=duration,
        loop=0,
    )
