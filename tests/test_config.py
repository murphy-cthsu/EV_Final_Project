from pathlib import Path

from motionprior.configs import load_config, REPO_CONFIG_DIR


def test_default_config_loads():
    cfg = load_config("default")
    assert cfg["gating"]["alpha0"] > 0
    assert isinstance(cfg["frequency_curriculum"]["milestones"], list)
    assert isinstance(cfg["articulation"]["lambda_intra"], float)


def test_named_config_resolves():
    cfg = load_config("dnerf_jumpingjacks")
    assert cfg["scene"]["name"] == "jumpingjacks"


def test_explicit_path_loads():
    cfg_path = Path(REPO_CONFIG_DIR) / "default.yaml"
    cfg = load_config(str(cfg_path))
    assert cfg["gating"]["alpha0"] > 0
