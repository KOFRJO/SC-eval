import os
import pytest

import numpy as np
from fd_shifts.analysis import metrics


@pytest.fixture
def mock_env_if_missing(monkeypatch) -> None:
    monkeypatch.setenv(
        "EXPERIMENT_ROOT_DIR", os.getenv("EXPERIMENT_ROOT_DIR", default="./experiments")
    )
    monkeypatch.setenv(
        "DATASET_ROOT_DIR", os.getenv("DATASET_ROOT_DIR", default="./data")
    )


class SC_test(metrics.StatsCache):
    """Using AURC_DISPLAY_SCALE=1 and n_bins=20 for testing."""
    AUC_DISPLAY_SCALE = 1

    def __init__(self, confids, correct):
        super().__init__(confids, correct, n_bins=20, legacy=False)


class SC_scale1000_test(metrics.StatsCache):
    """Using AURC_DISPLAY_SCALE=1000 and n_bins=20."""
    AUC_DISPLAY_SCALE = 1000

    def __init__(self, confids, correct):
        super().__init__(confids, correct, n_bins=20, legacy=False)
