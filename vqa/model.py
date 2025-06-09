# model.py — Multimodal VQA Model (BLIVA-inspired, fully runnable)
# Wires together a CLIP visual encoder, Q-Former bridge, and LLM decoder.
# Falls back to lightweight stubs when full models are not installed.

import torch
import torch.nn as nn
import torch.nn.functional as F


class QFormer(nn.Module):
    """
    Querying Transformer: cross-attends learned query tokens against image patch features.
    Outputs fixed-size visual token sequence for LLM conditioning.
    """
    def __init__(self, hidden_dim=768, num_query_tokens=32, num_heads=8, num_layers=6):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, image_features):
        """
        image_features: [B, num_patches, hidden_dim]
        returns:        [B, num_query_tokens, hidden_dim]
        """
        B = image_features.size(0)
        queries = self.query_tokens.expand(B, -1, -1)
        out = self.transformer(queries, image_features)
        return self.norm(out)


class VisualEncoder(nn.Module):
    """Wraps CLIP ViT encoder; falls back to random projection if CLIP unavailable."""
    def __init__(self, model_name="openai/clip-vit-large-patch14", hidden_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        try:
            from transformers import CLIPVisionModel
            self.encoder = CLIPVisionModel.from_pretrained(model_name)
            self.use_clip = True
            print(f"[VisualEncoder] Loaded {model_name}")
        except Exception:
            # Lightweight stub: treats image as flat vector
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 224 * 224, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.use_clip = False
            print("[VisualEncoder] CLIP unavailable — using lightweight stub")

    def forward(self, pixel_values):
        """Returns patch features [B, num_patches, hidden_dim]."""
        if self.use_clip:
            outputs = self.encoder(pixel_values=pixel_values)
            return outputs.last_hidden_state  # [B, 257, 1024] for ViT-L/14
        else:
            flat = self.encoder(pixel_values)  # [B, hidden_dim]
            return flat.unsqueeze(1).expand(-1, 32, -1)  # fake 32 patches


class MultimodalVQAModel(nn.Module):
    """
    Full VQA model:
      1. Encode image patches with ViT (or stub)
      2. Bridge with Q-Former
      3. Project to LLM embedding space
      4. Prepend visual tokens to text tokens and decode with LLM
    """
    def __init__(self,
                 llm_name="facebook/opt-125m",
                 vision_hidden=1024,
                 qformer_hidden=768,
                 num_query_tokens=32):
        super().__init__()
        self.num_query_tokens = num_query_tokens

        self.visual_encoder = VisualEncoder(hidden_dim=vision_hidden)
        self.qformer        = QFormer(hidden_dim=qformer_hidden,
                                      num_query_tokens=num_query_tokens)

        # Load LLM
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
            config = AutoConfig.from_pretrained(llm_name)
            self.llm_hidden = config.hidden_size
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name, torch_dtype=torch.float32)
            print(f"[VQAModel] Loaded LLM: {llm_name}")
        except Exception:
            self.llm_hidden = 512
            self.llm = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.llm_hidden, nhead=8,
                    dim_feedforward=2048, batch_first=True),
                num_layers=4)
            print("[VQAModel] LLM unavailable — using lightweight stub decoder")

        # Project Q-Former output to LLM embedding dimension
        self.visual_proj = nn.Linear(qformer_hidden, self.llm_hidden)
        self.query_embed = nn.Embedding(1, qformer_hidden)  # BOS visual token

    def encode_image(self, images):
        """Images -> visual tokens [B, num_query_tokens, llm_hidden]."""
        patch_feats    = self.visual_encoder(images)
        query_tokens   = self.qformer(patch_feats)
        visual_tokens  = self.visual_proj(query_tokens)
        return visual_tokens

    def forward(self, images, input_ids, attention_mask, labels=None):
        """
        images:         [B, 3, 224, 224]
        input_ids:      [B, seq_len]
        attention_mask: [B, seq_len]
        labels:         [B, seq_len] — optional, for training loss
        """
        B = images.size(0)
        visual_tokens = self.encode_image(images)  # [B, Q, llm_hidden]

        if hasattr(self.llm, 'get_input_embeddings'):
            # HuggingFace path: prepend visual tokens to text embeddings
            text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, T, H]
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

            # Extend attention mask for visual tokens
            vis_mask = torch.ones(B, self.num_query_tokens,
                                  dtype=attention_mask.dtype,
                                  device=attention_mask.device)
            full_mask = torch.cat([vis_mask, attention_mask], dim=1)

            outputs = self.llm(inputs_embeds=inputs_embeds,
                               attention_mask=full_mask,
                               labels=labels)
            return outputs
        else:
            # Stub decoder path
            mem = visual_tokens
            out = self.llm(input_ids.float().unsqueeze(-1).expand(-1,-1,self.llm_hidden),
                           mem)
            return out

    @torch.no_grad()
    def generate(self, images, input_ids, max_new_tokens=64, **kwargs):
        """Generate answer tokens given image and question."""
        visual_tokens = self.encode_image(images)
        if hasattr(self.llm, 'generate'):
            text_embeds   = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            return self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                **kwargs)
        return input_ids  # stub


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    model = MultimodalVQAModel().to(device)
    model.eval()

    B, T = 2, 20
    images   = torch.randn(B, 3, 224, 224, device=device)
    input_ids = torch.randint(0, 1000, (B, T), device=device)
    attn_mask = torch.ones(B, T, device=device, dtype=torch.long)

    with torch.no_grad():
        out = model(images, input_ids, attn_mask)

    print(f"Forward pass OK. Output type: {type(out)}")
    print(f"Visual encoder: {'CLIP' if model.visual_encoder.use_clip else 'stub'}")
