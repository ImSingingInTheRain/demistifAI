from __future__ import annotations
from typing import Tuple
import numpy as np
import streamlit as st

# Expect encode_texts to be importable in your app environment.
from demistifai.core.utils import encode_texts  # or adapt import path

@st.cache_data(show_spinner=False)
def _compute_cached_embeddings(dataset_hash: str, texts: tuple[str, ...]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return encode_texts(list(texts))
