import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.explainability import explain_prediction

samples = [
    ("Scientists Discover Coffee Cures All Diseases Overnight", "Researchers reportedly found that drinking coffee three times a day eliminates all known illnesses within 48 hours. The report spread widely online despite a lack of supporting evidence.", "Health", "2024-01-05", 0),
    ("Regional Hospital Opens New Emergency Wing", "The hospital inaugurated a new emergency-care facility designed to reduce patient wait times and increase treatment capacity.", "Health", "2024-01-15", 1),
    ("Government To Replace Elections With Online Popularity Contests", "A viral article claimed voters would soon be replaced by a social-media ranking system for selecting public officials.", "Politics", "2024-02-12", 0),
    ("City Launches Public Transportation Improvement Program", "Municipal authorities announced additional bus routes and upgraded ticketing systems to improve commuter experience.", "Transportation", "2024-02-18", 1),
    ("Moon Researchers Reveal Underground Alien Shopping Mall", "Several blogs reported that scientists had discovered a large commercial complex beneath the lunar surface.", "Science", "2024-03-01", 0),
    ("University Publishes Findings On Water Conservation", "Researchers released a study describing methods to reduce urban water consumption through infrastructure upgrades.", "Science", "2024-03-06", 1),
]

for headline, body, category, date, expected in samples:
    text = headline + ' ' + body
    expected_label = 'FAKE' if expected == 0 else 'REAL'
    print(f'---\nTEXT={headline}\nEXPECTED={expected_label}')
    for model in [
        'TF-IDF + Logistic Regression (Linear Baseline)',
        'TF-IDF + Random Forest (Ensemble Tree)',
        'Fine-tuned DistilBERT (Transformer)'
    ]:
        res = explain_prediction(text, model)
        pred = res.get('prediction')
        conf = res.get('confidence', 0)
        print(f'MODEL={model} -> PRED={pred} CONF={conf}%')
    print()
