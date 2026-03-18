"""
Multi-Model Fraud Detection Engine
Implements: Isolation Forest, Random Forest, XGBoost, and Autoencoder (Deep Learning)
All models are trained, evaluated, and persisted for real-time inference.
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import xgboost as xgb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ─── Optional TF Import ───────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from fraud_detection.data_generator import generate_transaction_dataset, FEATURE_COLUMNS

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ISOLATION FOREST  (Unsupervised Anomaly Detection)
# ══════════════════════════════════════════════════════════════════════════════
class IsolationForestDetector:
    """Unsupervised anomaly detection – no labels needed during training."""

    def __init__(self, contamination: float = 0.02):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_features=0.8,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = RobustScaler()

    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        raw = self.model.predict(X_scaled)          # 1 = normal, -1 = anomaly
        return (raw == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)  # negate so higher = more suspicious
        # Normalise to [0, 1]
        mn, mx = scores.min(), scores.max()
        return (scores - mn) / (mx - mn + 1e-9)

    def save(self, path: str = "models/isolation_forest.pkl"):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    @classmethod
    def load(cls, path: str = "models/isolation_forest.pkl"):
        obj = cls()
        data = joblib.load(path)
        obj.model  = data['model']
        obj.scaler = data['scaler']
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# 2. RANDOM FOREST  (Supervised Classification)
# ══════════════════════════════════════════════════════════════════════════════
class RandomForestDetector:
    """Supervised fraud classifier with built-in feature importance."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # SMOTE to balance classes
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X, y)
        X_scaled = self.scaler.fit_transform(X_res)
        self.model.fit(X_scaled, y_res)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def save(self, path: str = "models/random_forest.pkl"):
        joblib.dump({'model': self.model, 'scaler': self.scaler,
                     'fi': self.feature_importances_}, path)

    @classmethod
    def load(cls, path: str = "models/random_forest.pkl"):
        obj = cls()
        data = joblib.load(path)
        obj.model  = data['model']
        obj.scaler = data['scaler']
        obj.feature_importances_ = data['fi']
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# 3. XGBOOST  (Gradient Boosted Trees)
# ══════════════════════════════════════════════════════════════════════════════
class XGBoostDetector:
    """High-performance gradient-boosted fraud detector."""

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=50,   # Account for class imbalance
            eval_metric='aucpr',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        self.scaler = StandardScaler()
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X)
        eval_set = [(X_scaled, y)]
        if X_val is not None:
            eval_set.append((self.scaler.transform(X_val), y_val))
        self.model.fit(X_scaled, y, eval_set=eval_set, verbose=False)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def save(self, path: str = "models/xgboost.pkl"):
        joblib.dump({'model': self.model, 'scaler': self.scaler,
                     'fi': self.feature_importances_}, path)

    @classmethod
    def load(cls, path: str = "models/xgboost.pkl"):
        obj = cls()
        data = joblib.load(path)
        obj.model  = data['model']
        obj.scaler = data['scaler']
        obj.feature_importances_ = data['fi']
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# 4. AUTOENCODER  (Deep Learning Anomaly Detection)
# ══════════════════════════════════════════════════════════════════════════════
class AutoencoderDetector:
    """
    Deep Autoencoder trained ONLY on legitimate transactions.
    Fraud shows up as high reconstruction error.
    """

    def __init__(self, input_dim: int = 20):
        self.input_dim = input_dim
        self.threshold = None
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

    def _build_model(self) -> "keras.Model":
        inp = keras.Input(shape=(self.input_dim,))
        # Encoder
        x = layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(1e-4))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        encoded = layers.Dense(16, activation='relu', name='bottleneck')(x)
        # Decoder
        x = layers.Dense(32, activation='relu')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        decoded = layers.Dense(self.input_dim, activation='linear')(x)
        model = keras.Model(inp, decoded, name='Autoencoder')
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
        return model

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_legit = X[y == 0]
        X_scaled = self.scaler.fit_transform(X_legit)
        self.model = self._build_model()
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        self.history = self.model.fit(
            X_scaled, X_scaled,
            epochs=100, batch_size=256,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0
        )
        # Determine threshold on legit reconstruction errors (95th percentile)
        recon = self.model.predict(X_scaled, verbose=0)
        errors = np.mean(np.square(X_scaled - recon), axis=1)
        self.threshold = np.percentile(errors, 95)
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        recon = self.model.predict(X_scaled, verbose=0)
        return np.mean(np.square(X_scaled - recon), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        errors = self.reconstruction_error(X)
        return (errors > self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        errors = self.reconstruction_error(X)
        mn, mx = 0, self.threshold * 3
        proba = np.clip((errors - mn) / (mx - mn + 1e-9), 0, 1)
        return proba

    def save(self, path_prefix: str = "models/autoencoder"):
        self.model.save(f"{path_prefix}_model.keras")
        joblib.dump({'scaler': self.scaler, 'threshold': self.threshold,
                     'input_dim': self.input_dim}, f"{path_prefix}_meta.pkl")

    @classmethod
    def load(cls, path_prefix: str = "models/autoencoder"):
        meta = joblib.load(f"{path_prefix}_meta.pkl")
        obj = cls(input_dim=meta['input_dim'])
        obj.scaler    = meta['scaler']
        obj.threshold = meta['threshold']
        obj.model = keras.models.load_model(f"{path_prefix}_model.keras")
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE SCORER
# ══════════════════════════════════════════════════════════════════════════════
class EnsembleDetector:
    """
    Weighted ensemble combining all four detectors.
    Final score = weighted average of individual fraud probabilities.
    """

    WEIGHTS = {
        'isolation_forest': 0.15,
        'random_forest':    0.30,
        'xgboost':          0.40,
        'autoencoder':      0.15,
    }

    def __init__(self):
        self.iso   = None
        self.rf    = None
        self.xgb   = None
        self.ae    = None

    def predict_ensemble(self, X: np.ndarray) -> dict:
        scores = {}
        scores['isolation_forest'] = self.iso.score_samples(X) if self.iso else np.zeros(len(X))
        scores['random_forest']    = self.rf.predict_proba(X)  if self.rf  else np.zeros(len(X))
        scores['xgboost']          = self.xgb.predict_proba(X) if self.xgb else np.zeros(len(X))
        scores['autoencoder']      = self.ae.predict_proba(X)  if self.ae  else np.zeros(len(X))

        ensemble = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        predictions = (ensemble >= 0.5).astype(int)
        return {
            'scores': scores,
            'ensemble_score': ensemble,
            'predictions': predictions
        }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(name: str, y_true, y_pred, y_score=None) -> dict:
    """Compute and print classification metrics."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        'name':      name,
        'precision': report['1']['precision'],
        'recall':    report['1']['recall'],
        'f1_score':  report['1']['f1-score'],
        'accuracy':  report['accuracy'],
    }
    if y_score is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_score)
        metrics['avg_precision'] = average_precision_score(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Avg Prec:  {metrics['avg_precision']:.4f}")
    print(f"  Confusion Matrix:\n  {cm}")
    return metrics


def train_all_models(n_samples: int = 15000) -> dict:
    """Full training pipeline for all four detectors."""
    print("=" * 60)
    print("  AI Fraud Detection System — Training Pipeline")
    print("=" * 60)

    # ─── Data Generation ──────────────────────────────────────────
    print("\n[1/6] Generating synthetic transaction data …")
    df = generate_transaction_dataset(n_samples=n_samples, fraud_ratio=0.025)
    X  = df[FEATURE_COLUMNS].values
    y  = df['is_fraud'].values
    print(f"      Total: {len(df):,}  |  Fraud: {y.sum():,}  |  Legit: {(y==0).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    results = {}

    # ─── 1. Isolation Forest ──────────────────────────────────────
    print("\n[2/6] Training Isolation Forest …")
    t0 = time.time()
    iso = IsolationForestDetector(contamination=0.025)
    iso.fit(X_train)
    iso.save()
    preds = iso.predict(X_test)
    scores = iso.score_samples(X_test)
    results['isolation_forest'] = evaluate_model("Isolation Forest", y_test, preds, scores)
    results['isolation_forest']['train_time'] = round(time.time() - t0, 2)

    # ─── 2. Random Forest ─────────────────────────────────────────
    print("\n[3/6] Training Random Forest …")
    t0 = time.time()
    rf = RandomForestDetector()
    rf.fit(X_train, y_train)
    rf.save()
    preds  = rf.predict(X_test)
    scores = rf.predict_proba(X_test)
    results['random_forest'] = evaluate_model("Random Forest", y_test, preds, scores)
    results['random_forest']['train_time'] = round(time.time() - t0, 2)
    results['random_forest']['feature_importances'] = rf.feature_importances_.tolist()

    # ─── 3. XGBoost ───────────────────────────────────────────────
    print("\n[4/6] Training XGBoost …")
    t0 = time.time()
    xg = XGBoostDetector()
    xg.fit(X_train, y_train, X_test, y_test)
    xg.save()
    preds  = xg.predict(X_test)
    scores = xg.predict_proba(X_test)
    results['xgboost'] = evaluate_model("XGBoost", y_test, preds, scores)
    results['xgboost']['train_time']           = round(time.time() - t0, 2)
    results['xgboost']['feature_importances']  = xg.feature_importances_.tolist()

    # ─── 4. Autoencoder ───────────────────────────────────────────
    if TF_AVAILABLE:
        print("\n[5/6] Training Autoencoder …")
        t0 = time.time()
        ae = AutoencoderDetector(input_dim=X_train.shape[1])
        ae.fit(X_train, y_train)
        ae.save()
        preds  = ae.predict(X_test)
        scores = ae.predict_proba(X_test)
        results['autoencoder'] = evaluate_model("Autoencoder", y_test, preds, scores)
        results['autoencoder']['train_time'] = round(time.time() - t0, 2)
    else:
        print("\n[5/6] TensorFlow not available — skipping Autoencoder")

    # ─── Save metrics ─────────────────────────────────────────────
    print("\n[6/6] Saving training results …")
    with open("models/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n✅  All models trained and saved to models/")
    print("=" * 60)
    return results


if __name__ == "__main__":
    train_all_models()
