# ============================================================
# PRUDENCIA
# Configuration globale
# ============================================================

from pathlib import Path

# -------------------------------------------------------------------
# Répertoires du projet
# -------------------------------------------------------------------

# Racine de l'application (app/)
APP_DIR = Path(__file__).resolve().parent

# Racine du projet
PROJECT_DIR = APP_DIR.parent

# Données
DATA_DIR = PROJECT_DIR / "data"

# Modèles
MODELS_DIR = PROJECT_DIR / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
FINE_TUNED_DIR = MODELS_DIR / "fine_tuned"

# Datasets
DATASETS_DIR = PROJECT_DIR / "datasets"

# Logs
LOGS_DIR = PROJECT_DIR / "logs"

# Exports
EXPORTS_DIR = PROJECT_DIR / "exports"

# -------------------------------------------------------------------
# Modèles disponibles
# -------------------------------------------------------------------

AVAILABLE_MODELS = {
    "camembert": {
        "name": "CamemBERT Base",
        "hf_id": "almanach/camembert-base",
        "folder": "camembert",
    },
    "camembertv2": {
        "name": "CamemBERTv2 Base",
        "hf_id": "almanach/camembertv2-base",
        "folder": "camembertv2",
    },
    "juribert": {
        "name": "JuriBERT Base",
        "hf_id": "dascim/juribert-base",
        "folder": "juribert",
    },
}

# -------------------------------------------------------------------
# Paramètres par défaut du Fine-Tuning
# -------------------------------------------------------------------

DEFAULT_TRAINING = {
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "max_length": 512,
    "weight_decay": 0.01,
    "test_size": 0.2,
    "random_state": 42,
}

# -------------------------------------------------------------------
# Création automatique des dossiers
# -------------------------------------------------------------------

for directory in [
    MODELS_DIR,
    PRETRAINED_DIR,
    FINE_TUNED_DIR,
    DATASETS_DIR,
    LOGS_DIR,
    EXPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)