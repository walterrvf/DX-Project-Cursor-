import cv2
from PIL import Image, ImageTk
import numpy as np
from typing import Tuple, Optional


def cv2_to_tk(img_bgr: np.ndarray, max_w: Optional[int] = None, max_h: Optional[int] = None, scale_percent: Optional[float] = None) -> Tuple[Optional[ImageTk.PhotoImage], float]:
    """
    Converte imagem OpenCV BGR para PhotoImage do Tkinter, com redimensionamento.
    Retorna (photo_image, escala_aplicada).
    """
    if img_bgr is None or img_bgr.size == 0:
        return None, 1.0

    height, width = img_bgr.shape[:2]
    scale = 1.0

    if scale_percent is not None:
        scale = scale_percent / 100.0
    else:
        if max_w and max_h:
            scale_w = max_w / width
            scale_h = max_h / height
            scale = max(scale_w, scale_h)
        elif max_w:
            scale = max_w / width
        elif max_h:
            scale = max_h / height

    if scale != 1.0:
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        try:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            img_bgr_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interpolation)
        except cv2.error:
            return None, 1.0
    else:
        img_bgr_resized = img_bgr

    try:
        img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
        photo_image = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        return photo_image, scale
    except Exception:
        return None, scale


__all__ = ["cv2_to_tk"]


