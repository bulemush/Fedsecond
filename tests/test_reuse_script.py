"""Exercise the real Bash reuse stages without loading models or starting training."""

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fedproxy.config import load_config, save_resolved_config


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bash():
    candidates = [shutil.which("bash")]
    if os.name == "nt" and (git := shutil.which("git")):
        candidates.insert(0, str(Path(git).resolve().parents[1] / "bin/bash.exe"))
    for executable in candidates:
        if executable and Path(executable).is_file():
            return executable
    pytest.skip("Bash is unavailable")


def _bash(bash, code, *arguments):
    # MSYS system symlinks let Git Bash test chained links without Windows
    # administrator privileges. Linux uses normal native symlinks.
    env = dict(os.environ, TEST_PYTHON=Path(sys.executable).as_posix(),
               TEST_SRC=(PROJECT / "src").as_posix(), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if os.name == "nt":
        env["MSYS"] = "winsymlinks:sys"
        env["MSYS2_ENV_CONV_EXCL"] = "PYTHONPATH"
        env["TEST_STRIP_CR"] = "1"
    return subprocess.run([bash, "-c", code, "reuse-test", *map(str, arguments)],
                          env=env, cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", timeout=30)


def _fixture(tmp_path, bash, destination):
    source = tmp_path / "baseline_arc_easy"
    shared = tmp_path / "baseline_obqa" / "compression"
    run = tmp_path / "MA_arc_easy"
    source.mkdir()
    (shared / "proxy").mkdir(parents=True)
    (shared / "compression_manifest.json").write_text("{}", encoding="utf-8")
    (source / "data_manifest.json").write_text("{}", encoding="utf-8")
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints/keep.txt").write_text("existing checkpoint marker", encoding="utf-8")
    source_path, run_path = source.as_posix(), run.as_posix()
    if os.name == "nt":
        # Keep config paths in MSYS form; mixing C:/... with its /tmp mount
        # alias would test Windows mount spelling instead of Linux symlinks.
        paths = _bash(bash, 'cygpath -u -- "$1"\ncygpath -u -- "$2"', source_path, run_path)
        assert paths.returncode == 0, paths.stderr
        source_path, run_path = paths.stdout.splitlines()
    cfg = load_config(PROJECT / "configs/ma/homogeneous/arc_easy.yaml")
    cfg["run"].update(output_dir=run_path, reuse_artifacts_from=source_path,
                      reuse_compression_from=source_path)
    config = tmp_path / "experiment.yaml"
    save_resolved_config(cfg, config)
    original = copy.deepcopy(cfg)
    original["ma"]["enabled"] = False
    for command in ("train", "compress"):
        save_resolved_config(original, source / f"resolved_config.{command}.yaml")
    link = _bash(bash, 'ln -s -- "$1" "$2"\n[[ -L "$2" ]]', shared.as_posix(), (source / "compression").as_posix())
    if link.returncode:
        pytest.skip(f"Bash symlinks unavailable: {link.stderr}")
    if destination != "missing":
        target = source / "compression" if destination == "chained" else shared
        if destination == "wrong":
            target = tmp_path / "unrelated_compression"
            target.mkdir()
        link = _bash(bash, 'ln -s -- "$1" "$2"', target.as_posix(), (run / "compression").as_posix())
        assert link.returncode == 0, link.stderr
    return config, run


def _reuse(bash, config, stage):
    # Config reads and fedproxy.reuse run normally. Only GPU CLI stages are
    # stubbed, so these tests never import torch or execute real training.
    code = '''
python() {
  if [[ "$1" == "-m" && "$2" == "fedproxy.cli" ]]; then
    printf 'stub_cli=%s\\n' "$3"
    return 0
  fi
  if [[ -n "${TEST_STRIP_CR:-}" ]]; then
    PYTHONPATH="$TEST_SRC" "$TEST_PYTHON" "$@" | tr -d '\\r'
    return "${PIPESTATUS[0]}"
  fi
  PYTHONPATH="$TEST_SRC" "$TEST_PYTHON" "$@"
}
export -f python
bash "$1" "$2" 2,3 "$3"
'''
    return _bash(bash, code, (PROJECT / "scripts/run_experiment.sh").as_posix(), config.as_posix(), stage)


@pytest.mark.parametrize("stage", ["reuse-audit", "reuse-compression-all"])
@pytest.mark.parametrize("destination", ["missing", "chained", "direct"])
def test_reuse_accepts_chained_source_and_is_repeatable(tmp_path, bash, stage, destination):
    config, run = _fixture(tmp_path, bash, destination)
    for _ in range(2):
        result = _reuse(bash, config, stage)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "stub_cli=audit-truncation" in result.stdout
    assert (run / "checkpoints/keep.txt").read_text(encoding="utf-8") == "existing checkpoint marker"


@pytest.mark.parametrize("stage", ["reuse-audit", "reuse-compression-all"])
def test_reuse_still_rejects_link_to_unrelated_artifact(tmp_path, bash, stage):
    config, run = _fixture(tmp_path, bash, "wrong")
    result = _reuse(bash, config, stage)
    assert result.returncode == 2
    assert "link points elsewhere" in result.stderr
    assert "stub_cli=train" not in result.stdout
    assert (run / "checkpoints/keep.txt").read_text(encoding="utf-8") == "existing checkpoint marker"
