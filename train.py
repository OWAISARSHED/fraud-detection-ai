"""
Standalone training script.
Run this ONCE before starting the API server to pre-train all models.
Usage: python train.py
"""

from fraud_detection.models import train_all_models

if __name__ == "__main__":
    print("Starting FraudShield AI Model Training Pipeline...")
    results = train_all_models(n_samples=15000)
    print("\n📊 Summary:")
    for model, metrics in results.items():
        f1  = metrics.get('f1_score', 0)
        auc = metrics.get('roc_auc', 0)
        print(f"  {model:25s} F1={f1:.4f}  AUC={auc:.4f}")
    print("\n✅ Ready! Start the server with: python -m uvicorn app:app --reload")
