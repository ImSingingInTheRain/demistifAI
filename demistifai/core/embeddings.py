from __future__ import annotations
import numpy as np
import streamlit as st

from demistifai.modeling import encode_texts

@st.cache_data(show_spinner=False)
def _compute_cached_embeddings(dataset_hash: str, texts: tuple[str, ...]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return encode_texts(list(texts))
