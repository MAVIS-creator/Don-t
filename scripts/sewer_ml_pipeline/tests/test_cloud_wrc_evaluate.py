import numpy as np
import pytest

from scripts.sewer_ml_pipeline.cloud_wrc_evaluate import parse_rows, truth
from scripts.sewer_ml_pipeline.cloud_wrc_train import (
    class_positive_weights,
    configure_controlled_unfreeze,
    require_gpu,
    should_unfreeze_backbone_layer,
)


def test_parse_rows_and_truth_follow_target_order():
    text = "Filename,GCS_URI,FS,IS,RB,DE,RO,AF,BE,FO,ND\nframe.jpg,gs://bucket/frame.jpg,1,0,1,0,0,0,0,0,1\n"
    rows = parse_rows(text)
    assert rows[0]["GCS_URI"] == "gs://bucket/frame.jpg"
    assert truth(rows).tolist() == [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]


def test_parse_rows_rejects_missing_labels():
    with pytest.raises(ValueError, match="missing columns"):
        parse_rows("Filename,GCS_URI,FS\nframe.jpg,gs://bucket/frame.jpg,1\n")


def test_truth_is_binary_float_matrix():
    text = "Filename,GCS_URI,FS,IS,RB,DE,RO,AF,BE,FO,ND\na.jpg,gs://b/a,1.0,0,0,0,0,0,0,0,0\n"
    values = truth(parse_rows(text))
    assert values.dtype == np.float32
    assert values.shape == (1, 9)


def test_class_positive_weights_upweight_rare_labels():
    text = (
        "Filename,GCS_URI,FS,IS,RB,DE,RO,AF,BE,FO,ND\n"
        "a.jpg,gs://b/a,1,1,0,0,0,0,0,0,0\n"
        "b.jpg,gs://b/b,1,0,0,0,0,0,0,0,0\n"
        "c.jpg,gs://b/c,1,0,0,0,0,0,0,0,0\n"
        "d.jpg,gs://b/d,0,0,0,0,0,0,0,0,0\n"
    )
    weights = class_positive_weights(parse_rows(text))
    assert weights[1] > weights[0]
    assert weights[2] == 4.0


def test_controlled_unfreezing_targets_only_top_backbone_blocks():
    assert should_unfreeze_backbone_layer("block8_10_mixed")
    assert should_unfreeze_backbone_layer("conv_7b")
    assert not should_unfreeze_backbone_layer("block7a_branch_0")


class FakeLayer:
    def __init__(self, name):
        self.name = name
        self.trainable = False


class FakeBatchNormalization(FakeLayer):
    pass


def test_controlled_unfreeze_supports_flattened_backbone():
    layers = [
        FakeLayer("block17_20_ac"),
        FakeLayer("block8_10_mixed"),
        FakeBatchNormalization("conv_7b_bn"),
        FakeLayer("conv_7b"),
        FakeLayer("sewerml_dense_logits"),
    ]
    model = type("Model", (), {"layers": layers})()

    selected = configure_controlled_unfreeze(model, FakeBatchNormalization)

    assert selected == ["block8_10_mixed", "conv_7b"]
    assert [layer.trainable for layer in layers] == [False, True, False, True, True]


def test_require_gpu_fails_closed_without_visible_gpu():
    config = type("Config", (), {"list_physical_devices": staticmethod(lambda kind: [])})()
    tensorflow = type("TensorFlow", (), {"config": config})()

    with pytest.raises(RuntimeError, match="requires a visible GPU"):
        require_gpu(tensorflow)
