import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train_logistic import run_pipeline as run_logistic
from src.train_random_forest import run_pipeline as run_random_forest

print('Training logistic regression with updated TF-IDF rules...')
log_metrics = run_logistic()
print('Logistic metrics:', log_metrics['accuracy'])

print('Training random forest with updated TF-IDF rules...')
rf_metrics = run_random_forest()
print('Random forest metrics:', rf_metrics['accuracy'])
