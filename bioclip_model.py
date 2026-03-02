from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image


DEFAULT_MODEL_ID = 'hf-hub:imageomics/bioclip'


_project_root = Path(__file__).resolve().parent
_default_hf_home = _project_root / 'models' / 'hf_cache'
os.environ.setdefault('HF_HOME', str(_default_hf_home))
_default_hf_home.mkdir(parents=True, exist_ok=True)

if os.getenv('BIOCLIP_OFFLINE', '0') == '1':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')


def get_model_id() -> str:
    return os.getenv('BIOCLIP_MODEL_ID', DEFAULT_MODEL_ID)


def select_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_bioclip_model(device: str | None = None):
    target_device = device or select_device()
    model_id = get_model_id()
    model, preprocess = open_clip.create_model_from_pretrained(model_id)
    model = model.to(target_device)
    model.eval()
    return model, preprocess, target_device


def encode_image(image: Image.Image, model, preprocess, device: str) -> np.ndarray:
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
    return embedding.detach().cpu().numpy().astype('float32')
