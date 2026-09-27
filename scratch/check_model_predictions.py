import sys
from pathlib import Path

sys.path.insert(0, r'C:\Users\Admin\Desktop\Fake')

from src.explainability import explain_prediction

samples = [
    (
        'Real-like',
        'The World Health Organization said a new study published in The Lancet found that vaccination reduced severe respiratory illness across 18 countries, with health officials reporting lower emergency admissions after the program expanded this winter.'
    ),
    (
        'Fake-like',
        'SHOCKING! Doctors are hiding a secret cure that makes every disease vanish instantly and the government has been suppressing the truth for years. Share this before the media deletes the video.'
    ),
    (
        'Neutral',
        'City officials said a public meeting will be held next week to review traffic plans and discuss repairs to a bridge near the downtown bus terminal.'
    ),
]

models = [
    'TF-IDF + Logistic Regression (Linear Baseline)',
    'TF-IDF + Random Forest (Ensemble Tree)',
    'Fine-tuned DistilBERT (Transformer)',
]

lines = []
for label, text in samples:
    lines.append(f'--- {label} ---')
    for model in models:
        result = explain_prediction(text, model_choice=model)
        if isinstance(result, dict) and 'prediction' in result:
            prob = result.get('probabilities', {})
            lines.append(
                f"{model} -> {result['prediction']} confidence={round(result.get('confidence', 0), 2)} "
                f"REAL={prob.get('REAL', 0)} FAKE={prob.get('FAKE', 0)}"
            )
        else:
            lines.append(f'{model} -> ERROR {result}')
    lines.append('')

out = Path(r'C:\Users\Admin\Desktop\Fake\scratch\prediction_check.txt')
out.write_text('\n'.join(lines), encoding='utf-8')
print('WROTE', out)
