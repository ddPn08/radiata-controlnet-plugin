import cv2
import numpy as np
import PIL.Image
from controlnet_aux import OpenposeDetector

from modules.shared import hf_diffusers_cache_dir

preprocessor_model = None


def preprocess(image: PIL.Image, preprocessor: str):
    if preprocessor == "canny":
        return canny(image)
    elif preprocessor == "openpose":
        return openpose(image)


def canny(image: PIL.Image):
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return PIL.Image.fromarray(image)


def openpose(image: PIL.Image):
    global preprocessor_model
    if preprocessor_model is None or type(preprocessor_model) != OpenposeDetector:
        del preprocessor_model
        preprocessor_model = OpenposeDetector.from_pretrained(
            "lllyasviel/ControlNet",
            cache_dir=hf_diffusers_cache_dir(),
        )
    return preprocessor_model(image)
