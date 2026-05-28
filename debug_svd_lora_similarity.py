import os
import sys
import tempfile
import types

import numpy as np

datasets_stub = types.ModuleType("datasets")
datasets_stub.load_dataset = None
datasets_stub.DatasetDict = dict
sys.modules.setdefault("datasets", datasets_stub)

from utils import compute_svd_lora_similarity, load_svd_lora_adapter


def plain_b_cosine(B_a, B_b, eps=1e-8):
    numerator = np.sum(B_a * B_b)
    denom_a = np.sum(B_a ** 2)
    denom_b = np.sum(B_b ** 2)
    return numerator / (np.sqrt(denom_a) * np.sqrt(denom_b) + eps)


def main():
    B = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    e = np.array([[0.5, 2.0]], dtype=np.float32)

    identical = compute_svd_lora_similarity({"B": B, "e": e}, {"B": B, "e": e})
    assert np.isclose(identical, 1.0, atol=1e-6), "Identical B/e adapters should have similarity 1"

    zero_e = np.zeros_like(e)
    zero_similarity = compute_svd_lora_similarity({"B": B, "e": zero_e}, {"B": B, "e": e})
    assert np.isfinite(zero_similarity), "Zero e should not create NaN/Inf"
    assert np.isclose(zero_similarity, 0.0, atol=1e-6), "Zero e should give similarity 0"

    with tempfile.TemporaryDirectory() as temp_dir:
        np.save(os.path.join(temp_dir, "query_lora_B_client_0_0.npy"), B)
        loaded = load_svd_lora_adapter("query", 0, 0, temp_dir)
        assert loaded["e"].shape == (1, 2)
        assert np.allclose(loaded["e"], np.ones((1, 2), dtype=np.float32))

        fallback_similarity = compute_svd_lora_similarity(loaded, loaded)
        expected = plain_b_cosine(B[0], B[0])
        assert np.isclose(fallback_similarity, expected, atol=1e-6), "Missing e should fall back to plain B cosine"

    B_a = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    B_b = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, -1.0], [1.0, -1.0]],
        ],
        dtype=np.float32,
    )
    e_ones = np.ones((2, 2), dtype=np.float32)
    multi_layer = compute_svd_lora_similarity({"B": B_a, "e": e_ones}, {"B": B_b, "e": e_ones})
    assert np.isclose(multi_layer, 0.5, atol=1e-6), "Multi-layer similarity should average layer scores"

    print("SVD-LoRA similarity debug checks passed.")


if __name__ == "__main__":
    main()
