import torch

from pytorch_foundations.batches import make_dataloader


def test_make_dataloader_returns_expected_batch_shapes():
    features = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    labels = torch.arange(10, dtype=torch.long)

    loader = make_dataloader(features, labels, batch_size=3)

    batches = list(loader)
    assert len(batches) == 4
    assert batches[0][0].shape == (3, 2)
    assert sum(len(batch_labels) for _, batch_labels in batches) == 10


def test_shuffled_dataloader_is_reproducible_for_a_seed():
    features = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    labels = torch.arange(10, dtype=torch.long)

    first_order = torch.cat(
        [
            batch_labels
            for _, batch_labels in make_dataloader(
                features, labels, batch_size=2, shuffle=True, seed=7
            )
        ]
    )
    second_order = torch.cat(
        [
            batch_labels
            for _, batch_labels in make_dataloader(
                features, labels, batch_size=2, shuffle=True, seed=7
            )
        ]
    )

    assert torch.equal(first_order, second_order)


def test_make_dataloader_rejects_misaligned_inputs():
    with torch.no_grad():
        features = torch.zeros((3, 2))
        labels = torch.zeros(2, dtype=torch.long)

    try:
        make_dataloader(features, labels)
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("make_dataloader should reject misaligned inputs")
