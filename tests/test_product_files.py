from pathlib import Path


def test_product_docs_exist():
    for p in [
        "DEPLOYMENT.md",
        "MODEL_CARD.md",
        "DATA_CARD.md",
        "PRODUCT_OVERVIEW.md",
        "API_REFERENCE.md",
        "CHANGELOG.md",
        "Dockerfile",
        "docker-compose.yml",
        "render.yaml",
    ]:
        assert Path(p).exists(), f"Missing {p}"
