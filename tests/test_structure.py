from pathlib import Path


def test_structure_exists():
    for p in [
        "configs/default.yaml",
        "scripts/download_data.py",
        "scripts/build_dataset.py",
        "scripts/train.py",
        "src/models/dustriskformer.py",
    ]:
        assert Path(p).exists()
