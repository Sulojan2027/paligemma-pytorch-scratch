from typing import Dict, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

class PaliGemmaProcessor:
    
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, image_size: int, num_image_tokens: int):
        super().__init__()
        
        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] # For object detection
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ] # For Segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        
        self.tokenizer = tokenizer
        
    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        
        assert len(images) == 1 and len(text) == 1, f"Recieved {len(images)} images for {len(text)} text prompts"
        
        processed_images = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale = 1 / 255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD
        )
        
        # [B, C, H, W]
        image_batch = np.stack(processed_images, axis=0)
        image_tensors = torch.tensor(self.image_batch)
        
        
        
        
        
        
        
        
        
        
        