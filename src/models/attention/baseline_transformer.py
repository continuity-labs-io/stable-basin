import torch
import torch.nn as nn


class BaselineTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 10000,
        pos_embedding_scale: float = 0.02,
        ff_expansion_factor: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * pos_embedding_scale)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, 
            dim_feedforward=d_model * ff_expansion_factor,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            enable_nested_tensor=False
        )

    def forward(self, latent_x: torch.Tensor):
        batch, seq_len, d_model = latent_x.size()

        # Add positional embeddings
        x = latent_x + self.pos_embedding[:, :seq_len, :]

        # Create a causal mask to prevent looking into the future
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(latent_x.device)

        # Pass through transformer
        out = self.transformer(x, mask=causal_mask, is_causal=True)
        return out
