import joblib
import os

def load_model(model_path: str):
    """Load a saved model from disk"""
    return joblib.load(model_path)

def save_model(model, model_path: str):
    """Save a model to disk"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path} ✓")

def get_project_root() -> str:
    """Return absolute path to project root"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))