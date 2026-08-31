import torch

from pytorch_foundations.checkpoints import CheckpointMetadata, load_checkpoint, save_checkpoint
from pytorch_foundations.model import TurtleClassifier


def test_checkpoint_round_trip_restores_model_and_metadata(tmp_path):
    model = TurtleClassifier(feature_dim=4, n_classes=3)
    features = torch.randn((5, 4), generator=torch.Generator().manual_seed(151))
    expected_logits = model(features).detach()
    metadata = CheckpointMetadata(epoch=4, seed=157, feature_dim=4, n_classes=3)
    path = tmp_path / "model.pt"

    save_checkpoint(path, model, metadata)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    loaded_metadata = load_checkpoint(path, model)

    assert loaded_metadata == metadata
    assert torch.allclose(model(features), expected_logits)
    assert path.exists()
