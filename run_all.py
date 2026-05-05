import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

scripts = [
    "scripts/00_verify_data.py",
    "scripts/01a_data_acquisition_part1.py",
    "scripts/01_build_player_features.py",
    "scripts/02_eda_and_archetypes.py",
    "scripts/03_train_models.py",
    "scripts/04_make_report_assets.py",
    "scripts/05_update_actual_outputs_docs.py",
]

for script in scripts:
    print(f"\n=== Running {script} ===", flush=True)
    result = subprocess.run(
        [sys.executable, "-u", str(ROOT / script)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    print(result.stdout, flush=True)

    if result.returncode != 0:
        print(f"\nFailed at: {script}", flush=True)
        raise SystemExit(result.returncode)

print("\nAll project scripts completed.")
