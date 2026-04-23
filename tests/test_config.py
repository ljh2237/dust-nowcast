from src.utils.config import load_config


def test_config_load():
    cfg = load_config("configs/default.yaml")
    assert "project" in cfg
    assert "dataset" in cfg
