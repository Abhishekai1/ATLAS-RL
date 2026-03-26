"""
ATLAS-RL Kaggle Notebook Entry Point
=====================================
Run this cell in a Kaggle notebook (GPU P100/T4):

  !pip install -q faiss-cpu sentence-transformers datasets accelerate
  exec(open("kaggle_run.py").read())
"""

import subprocess, sys, os

# ── install dependencies if needed ───────────────────────────────────────────
def install_deps():
    pkgs = [
        "faiss-cpu",
        "sentence-transformers",
        "datasets",
        "accelerate",
    ]
    for p in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

# ── set working dir to script location ───────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

if __name__ == "__main__":
    install_deps()
    from run_experiment import main
    main()
