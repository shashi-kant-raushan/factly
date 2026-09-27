import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_traditional
from src.config import TFIDF_VECTORIZER_PATH, LOGISTIC_MODEL_PATH, RANDOM_FOREST_PATH, ID2LABEL

samples = [
    ("Scientists Discover Coffee Cures All Diseases Overnight", "Researchers reportedly found that drinking coffee three times a day eliminates all known illnesses within 48 hours. The report spread widely online despite a lack of supporting evidence."),
    ("Regional Hospital Opens New Emergency Wing", "The hospital inaugurated a new emergency-care facility designed to reduce patient wait times and increase treatment capacity."),
    ("Government To Replace Elections With Online Popularity Contests", "A viral article claimed voters would soon be replaced by a social-media ranking system for selecting public officials."),
    ("University Publishes Findings On Water Conservation", "Researchers released a study describing methods to reduce urban water consumption through infrastructure upgrades."),
]

vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
log_model = joblib.load(LOGISTIC_MODEL_PATH)
rf_model = joblib.load(RANDOM_FOREST_PATH)

for headline, body in samples:
    text = headline + ' ' + body
    cleaned = preprocess_traditional(text)
    X = vectorizer.transform([cleaned])
    print('---')
    print(headline)
    for name, model in [('Logistic', log_model), ('RandomForest', rf_model)]:
        p = model.predict_proba(X)[0]
        label_id = int(p.argmax())
        label = ID2LABEL[label_id]
        conf = float(p[label_id]) * 100
        print(name, label, round(conf, 2))
