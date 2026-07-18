"""code_entropy.infer_qparams: shape-collision disambiguation logic."""
from alis_dwq.code_entropy import infer_qparams


def test_config_disambiguates_collision():
    # wq last dim 8, sc last dim 2: (4,32) and (2,64) pack identically
    wq, sc = (16, 8), (16, 2)
    assert infer_qparams(wq, sc, "m.l", {"bits": 2, "group_size": 64}) == (2, 64)
    assert infer_qparams(wq, sc, "m.l", {"bits": 4, "group_size": 32}) == (4, 32)


def test_per_base_override_wins():
    wq, sc = (16, 8), (16, 2)
    qcfg = {"bits": 2, "group_size": 64,
            "m.special": {"bits": 4, "group_size": 32}}
    assert infer_qparams(wq, sc, "m.special", qcfg) == (4, 32)
    assert infer_qparams(wq, sc, "m.other", qcfg) == (2, 64)


def test_default_prefers_gs64():
    # no config: documented preference is mlx-lm's default gs=64
    assert infer_qparams((16, 8), (16, 2), "m.l", {}) == (2, 64)


def test_impossible_shapes_return_none():
    assert infer_qparams((16, 7), (16, 3), "m.l", {}) is None
