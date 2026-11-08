import subprocess
import sys

import tmu


def test_cli_version_matches_package():
    """Bounded smoke test: 'python -m tmu.cli --version' runs and reports the package version."""
    out = subprocess.run(
        [sys.executable, "-m", "tmu.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = out.stdout + out.stderr
    assert out.returncode == 0, combined
    assert "tmu-gpt2" in combined, combined
    assert tmu.__version__ in combined, f"CLI version != {tmu.__version__}: {combined}"
