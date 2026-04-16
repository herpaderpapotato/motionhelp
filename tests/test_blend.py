"""Tests for FunscriptTCN.blend()."""

import torch
import pytest
from src.models.tcn import FunscriptTCN


def _make_model(**kwargs) -> FunscriptTCN:
    defaults = dict(d_model=64, n_blocks=2, kernel_size=3, dropout=0.0,
                    n_persons=2, n_keypoints=17, embed_dim=128, flow_dim=32)
    defaults.update(kwargs)
    return FunscriptTCN(**defaults)


def test_blend_alpha_zero_equals_self():
    """alpha=0 should produce a model identical to self."""
    m1 = _make_model()
    m2 = _make_model()
    blended = m1.blend(m2, alpha=0.0)
    for p_self, p_blend in zip(m1.parameters(), blended.parameters()):
        assert torch.allclose(p_self, p_blend), "alpha=0 blend should equal self"


def test_blend_alpha_one_equals_other():
    """alpha=1 should produce a model identical to other."""
    m1 = _make_model()
    m2 = _make_model()
    blended = m1.blend(m2, alpha=1.0)
    for p_other, p_blend in zip(m2.parameters(), blended.parameters()):
        assert torch.allclose(p_other, p_blend), "alpha=1 blend should equal other"


def test_blend_alpha_half_is_midpoint():
    """alpha=0.5 should produce parameters exactly midway between self and other."""
    m1 = _make_model()
    m2 = _make_model()
    blended = m1.blend(m2, alpha=0.5)
    for p1, p2, pb in zip(m1.parameters(), m2.parameters(), blended.parameters()):
        expected = 0.5 * p1 + 0.5 * p2
        assert torch.allclose(pb, expected, atol=1e-6), "alpha=0.5 blend should be midpoint"


def test_blend_does_not_modify_originals():
    """Blending should not mutate either source model."""
    m1 = _make_model()
    m2 = _make_model()
    params1_before = [p.clone() for p in m1.parameters()]
    params2_before = [p.clone() for p in m2.parameters()]
    m1.blend(m2, alpha=0.3)
    for before, after in zip(params1_before, m1.parameters()):
        assert torch.equal(before, after), "m1 should not be modified"
    for before, after in zip(params2_before, m2.parameters()):
        assert torch.equal(before, after), "m2 should not be modified"


def test_blend_invalid_alpha_raises():
    m1 = _make_model()
    m2 = _make_model()
    with pytest.raises(ValueError):
        m1.blend(m2, alpha=-0.1)
    with pytest.raises(ValueError):
        m1.blend(m2, alpha=1.1)


def test_blend_output_shape():
    """Blended model should produce correct output shape."""
    m1 = _make_model()
    m2 = _make_model()
    blended = m1.blend(m2, alpha=0.5)
    blended.eval()
    B, T, N, K = 2, 16, 2, 17
    kp = torch.rand(B, T, N, K, 3)
    emb = torch.rand(B, T, N, 128)
    flow = torch.rand(B, T, 32)
    with torch.no_grad():
        out = blended(kp, emb, flow)
    assert out.shape == (B, 4, T), f"expected ({B}, 4, {T}), got {out.shape}"


def test_blend_with_ddl():
    """blend() should work correctly when use_ddl=True."""
    m1 = _make_model(use_ddl=True)
    m2 = _make_model(use_ddl=True)
    blended = m1.blend(m2, alpha=0.5)
    for p1, p2, pb in zip(m1.parameters(), m2.parameters(), blended.parameters()):
        expected = 0.5 * p1 + 0.5 * p2
        assert torch.allclose(pb, expected, atol=1e-6)
