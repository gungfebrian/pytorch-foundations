import pytest
import torch

from pytorch_foundations.preprocessing import FeatureScaler


def test_feature_scaler_fits_on_training_features_only():
    train_features = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    test_features = torch.tensor([[4.0, 8.0]])

    scaler = FeatureScaler.fit(train_features)

    normalized_train = scaler.transform(train_features)
    normalized_test = scaler.transform(test_features)

    assert torch.allclose(normalized_train.mean(dim=0), torch.zeros(2))
    assert torch.allclose(normalized_train.std(dim=0, unbiased=False), torch.ones(2))
    assert torch.allclose(normalized_test, torch.tensor([[3.0, 3.0]]))


def test_feature_scaler_keeps_constant_columns_finite():
    features = torch.tensor([[1.0, 2.0], [1.0, 5.0]])

    normalized = FeatureScaler.fit(features).transform(features)

    assert torch.allclose(normalized[:, 0], torch.zeros(2))
    assert torch.isfinite(normalized).all()


@pytest.mark.parametrize("features", [torch.ones(3), torch.ones((0, 2))])
def test_feature_scaler_rejects_invalid_feature_matrices(features):
    with pytest.raises(ValueError):
        FeatureScaler.fit(features)


def test_feature_scaler_rejects_transform_with_wrong_feature_count():
    scaler = FeatureScaler.fit(torch.ones((2, 3)))

    with pytest.raises(ValueError, match="feature dimension"):
        scaler.transform(torch.ones((2, 2)))
