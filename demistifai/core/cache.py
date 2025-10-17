from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Tuple

import pandas as pd
import streamlit as st

from demistifai.modeling import HybridEmbedFeatsLogReg, combine_text, features_matrix


def _ensure_dataframe(df: Any) -> pd.DataFrame:
    """Return a defensive copy of *df* coerced to a DataFrame."""

    if isinstance(df, pd.DataFrame):
        return df.copy(deep=True)
    if df is None:
        return pd.DataFrame()
    if isinstance(df, Mapping):
        return pd.DataFrame([df]).copy(deep=True)
    if isinstance(df, Iterable) and not isinstance(df, (str, bytes)):
        return pd.DataFrame(df).copy(deep=True)
    raise TypeError("Unsupported dataset payload for preparation cache")


@st.cache_data(show_spinner=False)
def cached_prepare(
    df: Any,
    params: dict,
    *,
    lib_versions: dict,
) -> pd.DataFrame:
    """Return a normalized training DataFrame.

    The cache key automatically incorporates the ``lib_versions`` mapping in
    addition to the provided dataframe and params. Changing library versions or
    preparation options therefore invalidates the cache.
    """

    params = dict(params or {})
    prepared = _ensure_dataframe(df)

    column_mapping = params.get("column_mapping")
    if isinstance(column_mapping, Mapping) and column_mapping:
        prepared = prepared.rename(columns=dict(column_mapping))

    required_columns: Iterable[str] = params.get(
        "required_columns",
        ("title", "body", "label"),
    )
    for column in required_columns:
        if column not in prepared.columns:
            prepared[column] = ""

    string_columns: Iterable[str] = params.get(
        "string_columns",
        ("title", "body", "label"),
    )
    strip_whitespace: bool = bool(params.get("strip_whitespace", True))
    for column in string_columns:
        if column in prepared.columns:
            prepared[column] = prepared[column].fillna("").astype(str)
            if strip_whitespace:
                prepared[column] = prepared[column].str.strip()

    dropna_subset: Iterable[str] | None = params.get("dropna")
    if dropna_subset:
        prepared = prepared.dropna(subset=list(dropna_subset))

    if params.get("drop_duplicates"):
        prepared = prepared.drop_duplicates(subset=params.get("duplicate_subset"))

    return prepared.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_features(
    prepared_df: pd.DataFrame,
    params: dict,
    *,
    lib_versions: dict,
) -> Tuple[Any, Any]:
    """Return cached feature payload and labels for training."""

    params = dict(params or {})
    title_column = params.get("title_column", "title")
    body_column = params.get("body_column", "body")
    target_column = params.get("target_column", "label")

    if title_column not in prepared_df or body_column not in prepared_df:
        raise KeyError("Prepared dataframe is missing title/body columns")
    if target_column not in prepared_df:
        raise KeyError("Prepared dataframe is missing the target column")

    titles = prepared_df[title_column].astype(str).tolist()
    bodies = prepared_df[body_column].astype(str).tolist()
    labels = prepared_df[target_column].astype(str).tolist()

    combined_texts = [combine_text(title, body) for title, body in zip(titles, bodies)]
    numeric_matrix = features_matrix(titles, bodies)

    feature_payload = {
        "titles": titles,
        "bodies": bodies,
        "texts": combined_texts,
        "numeric": numeric_matrix,
    }

    return feature_payload, labels


@st.cache_resource(show_spinner=True)
def cached_train(
    X_train: Any,
    y_train: Iterable[str],
    model_params: dict,
    *,
    lib_versions: dict,
) -> Any:
    """Train and cache a ``HybridEmbedFeatsLogReg`` instance."""

    if not isinstance(X_train, Mapping):
        raise TypeError("X_train must be a mapping with titles/bodies")

    titles = list(X_train.get("titles") or [])
    bodies = list(X_train.get("bodies") or [])
    if len(titles) != len(bodies):
        raise ValueError("Titles and bodies must have matching lengths")

    labels = list(y_train or [])
    if len(labels) != len(titles):
        raise ValueError("Label count must match training samples")

    params = dict(model_params or {})
    numeric_adjustments = params.pop("numeric_adjustments", None)

    model = HybridEmbedFeatsLogReg(**params)
    model = model.fit(titles, bodies, labels)

    if numeric_adjustments:
        model.apply_numeric_adjustments(numeric_adjustments)

    return model
