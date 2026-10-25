from pathlib import Path
import sys
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
sys.path.insert(0, str(Path(__file__).parents[1]))
from train_v3 import make_model, verify_partitions

def test_efficientnet_v3_head_is_9_multilabel_logits():
    model=make_model('efficientnet_b0',pretrained=False).eval()
    with torch.no_grad(): output=model(torch.zeros(1,3,64,64))
    assert output.shape==(1,9)

def test_convnext_tiny_v3_head_is_9_multilabel_logits():
    model=make_model('convnext_tiny',pretrained=False).eval()
    with torch.no_grad(): output=model(torch.zeros(1,3,64,64))
    assert output.shape==(1,9)

def test_group_overlap_across_train_and_tuning_disallowed():
    rows={'train':[{'Filename':'a','video_id':'sequence-one'}],
          'early_stop':[{'Filename':'b','video_id':'sequence-two'}],
          'threshold_tune':[{'Filename':'c','video_id':'sequence-one'}],
          'dev_report':[{'Filename':'d','video_id':'sequence-four'}]}
    with pytest.raises(ValueError):verify_partitions(rows)
