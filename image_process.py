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
    
)