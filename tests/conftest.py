from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def change_cwd(monkeypatch):
    """Change cwd to tests/ so relative paths like ../data/data.csv resolve correctly."""
    monkeypatch.chdir(Path(__file__).parent)
