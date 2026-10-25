"""Run the existing case pipeline with an explicit local UR5 model asset."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument("--ur5-model", type=Path, required=True)
args, remaining = parser.parse_known_args()
if not args.ur5_model.is_file():
    parser.error("UR5 model file does not exist")
import create_coppeliasim_scene
create_coppeliasim_scene.UR5_MODEL_CANDIDATES = (str(args.ur5_model.resolve()),)
import runpy
sys.argv = [sys.argv[0], *remaining]
runpy.run_module("run_case_pipeline", run_name="__main__")
