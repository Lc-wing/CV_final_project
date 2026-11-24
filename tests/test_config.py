from pathlib import Path
import pytest
import yaml
from src.config import ExperimentConfig, load_config, validate_controlled_configs


def test_load_all_three_experiment_configs():
    cfgs = [
        load_config(Path("configs/res_800.yaml")),
        load_config(Path("configs/res_480.yaml")),
        load_config(Path("configs/res_480_small_anchor.yaml")),
    ]
    assert [c.run_name for c in cfgs] == ["res_800", "res_480", "res_480_small_anchor"]
    assert cfgs[0].min_size == 800 and cfgs[0].max_size == 1333
    assert cfgs[1].min_size == 480 and cfgs[1].max_size == 800
    assert cfgs[2].min_size == 480 and cfgs[2].max_size == 800
    assert cfgs[0].anchor_sizes == (32, 64, 128, 256, 512)
    assert cfgs[1].anchor_sizes == (32, 64, 128, 256, 512)
    assert cfgs[2].anchor_sizes == (16, 32, 64, 128, 256)
    validate_controlled_configs(cfgs)


def test_run_name_must_match_filename_stem(tmp_path):
    bad = tmp_path / "wrong_name.yaml"
    bad.write_text(Path("configs/res_800.yaml").read_text())
    with pytest.raises(ValueError, match="run_name"):
        load_config(bad)


def test_validate_controlled_configs_rejects_seed_drift():
    c1 = load_config("configs/res_800.yaml")
    c2 = load_config("configs/res_480.yaml")
    c2_bad = ExperimentConfig(**{**c2.__dict__, "seed": 99})
    with pytest.raises(ValueError, match="seed"):
        validate_controlled_configs([c1, c2_bad])


@pytest.mark.parametrize(
    "bad_anchors",
    [
        [32, 64, 128, 256],
        [16, 32, 64, 128, 256, 512],
        [0, 32, 64, 128, 256],
        [-16, 32, 64, 128, 256],
        [16.5, 32, 64, 128, 256],
        [True, 32, 64, 128, 256],
    ],
)
def test_load_config_rejects_invalid_anchor_sizes(tmp_path, bad_anchors):
    raw = yaml.safe_load(Path("configs/res_480.yaml").read_text())
    raw["run_name"] = "bad_anchors"
    raw["anchor_sizes"] = bad_anchors
    cfg_path = tmp_path / "bad_anchors.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="anchor_sizes"):
        load_config(cfg_path)


def test_load_config_rejects_min_size_greater_than_max_size(tmp_path):
    raw = yaml.safe_load(Path("configs/res_480.yaml").read_text())
    raw["run_name"] = "bad_sizes"
    raw["min_size"] = 900
    raw["max_size"] = 800
    cfg_path = tmp_path / "bad_sizes.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="min_size"):
        load_config(cfg_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trainable_backbone_layers", 5),
        ("epochs", 10),
        ("batch_size", 16),
        ("lr", 0.001),
        ("weight_decay", 0.001),
        ("warmup_steps", 100),
        ("grad_clip_norm", 5.0),
        ("hflip_prob", 0.0),
        ("score_thresh", 0.3),
        ("iou_thresh", 0.7),
        ("aspect_ratios", (0.5, 1.0)),
        ("anchor_sizes", (32, 64, 128, 256)),
    ],
)
def test_validate_controlled_configs_rejects_controlled_field_drift(field, value):
    c1 = load_config("configs/res_800.yaml")
    c2 = load_config("configs/res_480.yaml")
    c2_bad = ExperimentConfig(**{**c2.__dict__, field: value})
    with pytest.raises(ValueError, match=field):
        validate_controlled_configs([c1, c2_bad])


def test_validate_controlled_configs_rejects_empty_list():
    with pytest.raises(ValueError, match="empty|at least one"):
        validate_controlled_configs([])
