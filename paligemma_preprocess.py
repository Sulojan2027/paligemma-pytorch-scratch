from typing import List
import numpy as np
from PIL import Image
import torch

from image_process import process_images

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    image_token,
    img_seq_len
):
    return f"{image_token * img_seq_len}{bos_token}{prefix_prompt}\n"

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
        
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1 / 255.0,
            mean = IMAGENET_STANDARD_MEAN,
            std = IMAGENET_STANDARD_STD
        )
        
        # [B, C, H, W]
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)
        
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_token=self.IMAGE_TOKEN,
                img_seq_len=self.image_seq_length
            )
            for prompt in text
        ]
        
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation
        )
        
        return_data = {"pixel_values": pixel_values, **inputs}
        
        return return_data
        
        
        
        
        
        
        
        
        