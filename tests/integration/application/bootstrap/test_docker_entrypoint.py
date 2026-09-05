from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.support.paths import locate_repo_root


ENTRYPOINT = locate_repo_root(Path(__file__)) / "docker" / "entrypoint.sh"


@pytest.mark.parametrize(
    ("auto_sync", "sync_exit", "expected_exit", "expected_stdout"),
    [
        (None, 0, 7, "two words\n"),
        ("0", 23, 7, "two words\n"),
        (None, 23, 23, ""),
    ],
    ids=["sync-and-exec", "skip-sync", "sync-failure-stops-command"],
)
def test_entrypoint_syncs_before_forwarding_command(
    tmp_path: Path,
    auto_sync: str | None,
    sync_exit: int,
    expected_exit: int,
    expected_stdout: str,
) -> None:
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/bin/bash\npwd > sync.log\nprintf '%s\\n' \"$@\" >> sync.log\n"
        f"exit {sync_exit}\n"
    )
    uv.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.defpath}"}
    if auto_sync is not None:
        env["AUTO_SYNC"] = auto_sync

    completed = subprocess.run(
        ["bash", str(ENTRYPOINT), "bash", "-c", 'printf "%s\\n" "$1"; exit 7', "--", "two words"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == expected_exit, completed.stderr
    assert completed.stdout == expected_stdout
    sync_log = tmp_path / "sync.log"
    if auto_sync == "0":
        assert not sync_log.exists()
    else:
        assert sync_log.read_text().splitlines() == [
            str(tmp_path), "sync", "--locked", "--extra", "cuda132",
        ]
