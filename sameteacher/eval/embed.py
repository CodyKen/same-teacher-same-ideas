"""Text embeddings. `tfidf` needs no download and is also the lexical comparison.

SAMETEACHER_EMBED_DEVICE (for example `cpu`) overrides the device sentence-transformers picks,
for machines where the MPS memory pool runs out on a long run.
"""
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def embed_texts(texts: list[str], name: str, batch_size: int = 64):
    """Return unit-length row vectors: a dense array, or a sparse matrix for tfidf."""
    if name == "tfidf":
        matrix = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit_transform(texts)
        return normalize(matrix)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=os.environ.get("SAMETEACHER_EMBED_DEVICE") or None)
    return np.asarray(
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    )


def rows_dense(matrix, indices: list[int]) -> np.ndarray:
    block = matrix[indices]
    return block.toarray() if hasattr(block, "toarray") else block
