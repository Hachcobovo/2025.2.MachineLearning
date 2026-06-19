from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_random_forest():
    return RandomForestClassifier(
        n_estimators=100, 
        max_depth=None, 
        random_state=42, 
        n_jobs=-1
    )

def get_xgboost():
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        eval_metric='logloss',
        early_stopping_rounds=20,
        tree_method='hist',
        use_label_encoder=False,
        random_state=42
    )