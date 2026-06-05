import torch
from xmatcher.core.types import MatchResult, DenseField


def test_match_result_default_dense_is_none():
    r = MatchResult(
        mkpts0=torch.zeros(0, 2), mkpts1=torch.zeros(0, 2), mconf=torch.zeros(0),
        method="x", pair_id="y", runtime_ms=0.0,
    )
    assert r.dense is None


def test_dense_field_default_coord_space_is_processed():
    d = DenseField(warp=torch.zeros(2, 2, 4), certainty=torch.zeros(2, 2))
    assert d.coord_space == "processed"
