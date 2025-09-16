from typing import List, Union, Optional
import pathlib

try:
    import cv2

    # import os

    # os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
except:
    import logging

    logging.debug("cv2 not availble. Most functionality in images.py will break.")


try:
    import numpy as np
except:
    import logging

    logging.debug("numpy not availble. Most functionality in images.py will break.")

try:
    from PIL import Image
except:
    import logging

    logging.debug("PIl not availble. Most functionality in images.py will break.")


def save_images_to_gif(
    images: List["np.ndarray"],
    output_path: Union[pathlib.Path, str],
    fps: Optional[float] = None,
    duration: Optional[float] = None,
):
    assert not (duration is None and fps is None)

    if output_path.suffix == "":
        output_path = output_path.with_suffix(".gif")

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


def save_images_to_mp4(
    images: List["np.ndarray"],
    output_path: Union[pathlib.Path, str],
    fps: Optional[float] = None,
    duration: Optional[float] = None,
):
    assert not (duration is None and fps is None)

    if output_path.suffix == "":
        output_path = output_path.with_suffix(".avi")

    if fps is None:
        fps = len(images) / duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width, height, _ = images[0].shape
    video = cv2.VideoWriter(
        filename=str(output_path), fourcc=fourcc, fps=fps, frameSize=(height, width)
    )

    for image in images:
        # opencv expects BGR format --> convert RGB to BGR
        image = np.flip(image, axis=-1)  # Convert RGB to BGR

        # opencv expects uint8 images
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        assert image.dtype == np.uint8, "Image dtype must be uint8"


        video.write(image)

    # cv2.destroyAllWindows()
    video.release()
