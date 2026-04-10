import os
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_test_env.sh"


def _run_check_script(*args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [str(SCRIPT_PATH), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_check_test_env_uses_pytest_interpreter_and_passes_in_current_env():
    result = _run_check_script("--strict")

    assert result.returncode == 0, result.stderr
    assert "which python:" in result.stdout
    assert "python -V:" in result.stdout
    assert "sys.executable:" in result.stdout
    assert sys.executable in result.stdout


def test_check_test_env_soft_mode_warns_without_failing():
    result = _run_check_script("--soft", extra_env={"CHECK_TEST_ENV_REQUIRED_MODULES": "definitely_not_installed_mod"})

    assert result.returncode == 0
    assert "[check_test_env][WARN]" in result.stderr


def test_check_test_env_strict_mode_fails_on_missing_dependency():
    result = _run_check_script("--strict", extra_env={"CHECK_TEST_ENV_REQUIRED_MODULES": "definitely_not_installed_mod"})

    assert result.returncode == 1
    assert "[check_test_env][ERROR]" in result.stderr
