"""
Classical Machine Learning models wrapper module.

Provides unified interface for initializing, training, and predicting with:
- Logistic Regression
- Random Forest
- XGBoost
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_classical_model(model_name: str, config: Dict[str, Any]) -> Any:
    """
    Instantiates a classical ML model based on name and configuration dictionary.
    
    Parameters
    ----------
    model_name : str
        One of 'logistic_regression', 'random_forest', 'xgboost'.
    config : Dict[str, Any]
        Model configuration dictionary.
        
    Returns
    -------
    Any
        Unfitted scikit-learn or XGBoost model instance.
    """
    model_name = model_name.lower()
    
    if model_name == "logistic_regression":
        return LogisticRegression(
            solver=config.get("solver", "lbfgs"),
            max_iter=config.get("max_iter", 2000),
            random_state=config.get("random_state", 42),
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.get("n_estimators", 300),
            class_weight=config.get("class_weight", None),
            random_state=config.get("random_state", 42),
            n_jobs=config.get("n_jobs", -1),
        )
    elif model_name == "xgboost":
        return XGBClassifier(
            n_estimators=config.get("n_estimators", 300),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            objective=config.get("objective", "multi:softprob"),
            eval_metric=config.get("eval_metric", "mlogloss"),
            random_state=config.get("random_state", 42),
            n_jobs=config.get("n_jobs", -1),
        )
    else:
        raise ValueError(f"Unknown classical model name: {model_name}")


def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """
    Trains the given model on training features and integer targets.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (N, D).
    y_train : np.ndarray
        Training target labels of shape (N,).
        
    Returns
    -------
    Any
        Fitted model instance.
    """
    model.fit(X_train, y_train)
    return model


def predict_model(model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates class predictions and class probability distributions.
    
    Parameters
    ----------
    model : Any
        Fitted model instance.
    X : np.ndarray
        Feature matrix of shape (N, D).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (y_pred, y_prob) where y_pred has shape (N,) and y_prob has shape (N, K).
    """
    y_prob = model.predict_proba(X)
    
    # Handle edge case if model trained on subset of classes (e.g. 4 out of 5 classes in small test synthetic data)
    if y_prob.shape[1] < 5 and hasattr(model, "classes_"):
        full_prob = np.zeros((X.shape[0], 5), dtype=float)
        for i, cls in enumerate(model.classes_):
            if cls < 5:
                full_prob[:, int(cls)] = y_prob[:, i]
        y_prob = full_prob
        
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob
