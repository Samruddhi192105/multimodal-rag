"""
embeddings.py
Wraps a single CLIP model so that BOTH text chunks and images are embedded
into the same vector space. That shared space is what makes cross-modal
retrieval possible: a text query can pull back a relevant image, and vice
versa, because "distance" between a text vector and an image vector is
meaningful.
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import CLIP_MODEL_NAME


class ClipEmbedder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def _unwrap(features):
        """transformers>=5.0 wraps get_*_features() output in a
        BaseModelOutputWithPooling instead of returning a raw tensor.
        Handle both old (tensor) and new (wrapped) return types."""
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        return features

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Embed one or more text strings into CLIP's joint space."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        features = self._unwrap(self.model.get_text_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().tolist()

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed one or more PIL images into CLIP's joint space."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self._unwrap(self.model.get_image_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().tolist()

    def embed_single_text(self, text: str) -> list[float]:
        return self.embed_text([text])[0]

    def embed_single_image(self, image: Image.Image) -> list[float]:
        return self.embed_images([image])[0]
