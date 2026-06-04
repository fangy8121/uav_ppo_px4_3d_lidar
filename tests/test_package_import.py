import os
from pathlib import Path
import subprocess
import sys


def test_package_import_does_not_eagerly_load_gymnasium():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    source_path = str(project_root / "ros2_ws" / "src" / "uav_px4_rl")
    env["PYTHONPATH"] = os.pathsep.join(
        [source_path, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [source_path]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import uav_px4_rl; "
            "raise SystemExit(1 if 'gymnasium' in sys.modules else 0)",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
