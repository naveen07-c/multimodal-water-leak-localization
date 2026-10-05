import subprocess
import sys
from pathlib import Path

scripts = [
    r"models/baseline\src\training\trainer_full.py",
    r"models/baseline\eval_original_hk.py",
    r"models/hybrid_ml\train.py",
    r"models/transfer_cnn\train.py",
    r"models/safnet\train.py",
    r"scripts\compare_models.py"
]

import os

env = os.environ.copy()
env['PYTHONPATH'] = str(Path('.').absolute()) + os.pathsep + str(Path('models/baseline').absolute())

for script in scripts:
    print(f"\n{'='*50}\nRunning {script}...\n{'='*50}\n")
    result = subprocess.run([sys.executable, script], env=env)
    if result.returncode != 0:
        print(f"Error running {script}")
        break
    print(f"Finished {script}")

print("\nALL MODELS TRAINED AND EVALUATED!")
