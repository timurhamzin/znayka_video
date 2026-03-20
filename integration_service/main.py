from __future__ import annotations

import subprocess
from pathlib import Path


def run() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        'uv',
        'run',
        '--project',
        'integration_service',
        'uvicorn',
        'app.main:app',
        '--host',
        '127.0.0.1',
        '--port',
        '8010',
        '--reload',
    ]
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == '__main__':
    run()
