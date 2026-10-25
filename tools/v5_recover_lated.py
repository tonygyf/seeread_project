"""Run the recorded LATeD numerical repair without changing the active CTW closure."""
from pathlib import Path
import argparse
import runpy
import sys


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runtime', type=Path, default=root / 'results/v5/runtime/lated_geometry_fp32_20261025')
    args, remaining = parser.parse_known_args()
    runtime = args.runtime.resolve()
    script = runtime / 'tools/v5_train_detection.py'
    if not script.is_file():
        raise FileNotFoundError('Prepare the recorded LATeD runtime and healthy migration in docs/04 first')
    sys.path.insert(0, str(runtime))
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name='__main__')
