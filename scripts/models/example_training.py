#!/usr/bin/env python3
"""Example training pipeline and preprocessing visualization.

This script loads `outputs/model_ready.csv`, produces EDA plots into
`outputs/figures/`, trains a simple model (RandomForest if available,
otherwise a baseline), saves predictions and metrics.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import os


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
FIG_DIR = OUT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = OUT / 'model_ready.csv'


def try_import(pkg, alias=None):
    try:
        m = __import__(pkg)
        return m if alias is None else getattr(m, alias)
    except Exception:
        return None


def eda(df):
    # basic distribution of target
    if 'Total' in df.columns:
        if HAVE_MPL:
            plt.figure(figsize=(6,4))
            #!/usr/bin/env python3
            """Orchestrator: run data prepare -> plot -> train.

            This script calls the modular scripts in `scripts/data`, `scripts/plots`,
            and `scripts/models` to produce preprocessing figures and a trained model.
            """
            from pathlib import Path
            import importlib


            def main():
                # run data preparation
                mod = importlib.import_module('scripts.data.run_prepare')
                mod.main()

                # run plotting
                modp = importlib.import_module('scripts.plots.visualize_preprocessing')
                modp.main()

                # run model training
                modm = importlib.import_module('scripts.models.train_model')
                modm.main()


            if __name__ == '__main__':
                main()
            plt.title('Top 10 countries by total medals')
