import pytest
import torch

from pytorch_foundations.tensor_basics import TensorSummary, normalize_features, summarize_tensor


def test_summarize_tensor_reports_shape_dtype_ndim_and_numel():
    summary = summarize_tensor(torch.zeros((3, 4), dtype=torch.float32))

    assert summary == TensorSummary(shape=(3, 4), dtype="torch.float32", ndim=2, numel=12)


def test_normalize_features_centers_each_column():
    features = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])

    normalized = normalize_features(features)

    assert torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized.std(dim=0), torch.ones(2), atol=1e-6)


def test_normalize_features_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="2D"):
        normalize_features(torch.ones(3))
