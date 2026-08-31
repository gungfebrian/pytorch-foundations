from pytorch_foundations.model import TurtleClassifier
from pytorch_foundations.model_summary import ModelSummary, summarize_model


def test_summarize_model_counts_all_and_trainable_parameters():
    model = TurtleClassifier(feature_dim=24, n_classes=8)

    summary = summarize_model(model)

    assert summary == ModelSummary(parameter_count=1592, trainable_parameter_count=1592)
