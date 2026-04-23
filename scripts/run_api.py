from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

sys.path.append(str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    uvicorn.run("src.webapp.api:app", host="127.0.0.1", port=8000, reload=False)
