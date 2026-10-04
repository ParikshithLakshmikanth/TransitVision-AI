"""TransitVision AI - Pipeline Execution Script for Phase 5 Model Training.
Runs end-to-end model training, multi-algorithm benchmarking, error analysis,
and registers the production model.
"""
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.training.train_models import ModelTrainingPipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    pipeline = ModelTrainingPipeline()
    res = pipeline.run()
    print("\n--- MODEL TRAINING SUMMARY ---")
    print(f"Status: {res['status']}")
    print(f"Production Model: {res['selected_production_model']}")
    print(f"Artifact Directory: {res['artifact_dir']}")
    print("\nProduction Metrics:")
    for k, v in res['production_metrics'].items():
        print(f"  {k}: {v}")
