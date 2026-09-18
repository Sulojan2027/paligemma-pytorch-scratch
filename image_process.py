from typing import Dict, Optional, Union, Tuple, Iterable, List
from PIL import Image
import numpy as np

def resize_image(
    image: Image,
    size: Tuple[int, int],
    resampling: Image.Resampling= None,
    reduce_gap: Optional[int] = None
) -> np.ndarray:
    
    height, width = size
    resized_image = image.resize(
        (width, height),
        resample = resampling,
        reducing_gap = reduce_gap
    )
    
    return resized_image

def rescale_image(
    image: np.ndarray,
    scale: float,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    
    return rescaled_image

def normalize_image(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]]
) -> np.ndarray:
    
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    
    normalized_image = (image - mean) / std
    
    return normalized_image

def process_images(
    images: List[Image.Image],
    size: Dict[str, int]= None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    mean: Union[float, Iterable[float]] = None,
    std: Union[float, Iterable[float]] = None,
) -> List[np.ndarray]:
    
    height, width = size[0], size[1]
    
    # Resizing
    images = [resize_image(img, size=(height, width), resampling=resample) for img in images]
    # Rescaling
    images = [rescale_image(img, scale=rescale_factor) for img in images]
    # Normalizing
    images = [normalize_image(img, mean=mean, std=std) for img in images]
    
    images = [img.transpose(2, 0, 1) for img in images]
    
    return images
