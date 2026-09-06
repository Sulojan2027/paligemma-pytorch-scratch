from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size: int = 768,
        linear_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps=1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: int = None,
        **kwargs
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.linear_size = linear_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        assert (
            self.image_size // self.patch_size
        ), "Image size should be divisible by Patch size"

        # Patching and turning into embeddings
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "positional_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )
        
    def forward(self, image_batch: torch.FloatTensor) -> torch.Tensor:
        # [B, C, H, W]
        _, _, height, width = image_batch.shape
        
        # [B, C, H, W] -> [B, dim, H/P_s, W/P_s]
        patch_embeds = self.patch_embedding(image_batch)
        # [B, dim, H/P_s, W/P_s] -> [B, dim, N_p]
        embeddings = patch_embeds.flatten(2)
        # [B, dim, N_p] -> [B, N_p, dim]
        embeddings = embeddings.transpose(1, 2)
        
        final_embeddings = embeddings + self.position_embedding(self.positional_ids)
        
        return final_embeddings
    
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.linear_size)
        self.fc2 = nn.Linear(config.linear_size, config.hidden_size)
        
    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        fc1_embeds = self.fc1(embeds)
        fc1_embeds = F.gelu(fc1_embeds, approximate="tanh")
        fc2_embeds = self.fc2(fc1_embeds)
        return fc2_embeds
    
class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    
    
class SiglipVisionEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        residual = hidden_states
        norm1_embeds = self.layer_norm1(hidden_states)
        attn_embeds = self.self_attn(norm1_embeds)
        
        residual_embeds = attn_embeds + residual
        norm2_embeds = self.layer_norm2(residual_embeds)
        mlp_embeds = self.mlp(norm2_embeds)
        context_embeds = mlp_embeds + residual_embeds
        
        return context_embeds
        
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Image -> Embeddings
        self.embeddings = SiglipVisionEmbeddings(config)
        # Embeddings -> Encoder
        self.encoder = SiglipVisionEncoder(config)
        # Post Normalization
        self.post_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, image_pixels: torch.tensor) -> torch.tensor:
        embeds = self.embeddings(image_pixels)
        enc_embeds = self.encoder(input_embeds=embeds)
        norm_embeds = self.post_layer_norm(enc_embeds)

        return norm_embeds


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, image_pixels) -> Tuple:
        # [B, C, H, W] -> [B, N_p, dim]
        return self.vision_model(image_pixels=image_pixels)
