import os
import shutil

import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="The python called here for some reason will lack some dependencies in GH actions. Test locally instead.",
)
class TestMightCLI:
    def test_run_from_file(self):
        exit_status = os.system(
            "uv run python mighty/run_mighty.py num_steps=100 output_dir=test_cli"
        )
        assert exit_status == 0
        shutil.rmtree("test_cli")
