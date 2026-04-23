from src.utils.config import load_config


def test_default_region_is_cn_nw():
    cfg = load_config("configs/default.yaml")
    assert cfg["region"]["name"] == "cn_northwest_corridor"
