"""
Trains multiple ML models on the real dataset and saves the best one.
"""
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.preprocess import load_and_split

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    print("\n" + "=" * 60)
    print("   🎓 STUDENT PLACEMENT PREDICTION — MODEL TRAINING")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------
    # Load & preprocess real dataset
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test, scaler, used_features = load_and_split()
    print(f"\n📌 Features used ({len(used_features)}):")
    for f in used_features:
        print(f"   • {f}")
    print()

    # ------------------------------------------------------------
    # Define models
    # ------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200,
                                                      max_depth=12,
                                                      random_state=42),
    }

    best_model, best_acc, best_name = None, 0, ""

    # ------------------------------------------------------------
    # Train each model & evaluate
    # ------------------------------------------------------------
    print("=" * 60)
    print("       🎯 MODEL TRAINING RESULTS")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n📌 {name}")
        print(f"   Accuracy: {acc * 100:.2f}%")
        print(f"   Confusion Matrix: {confusion_matrix(y_test, y_pred).tolist()}")

        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name

    # ------------------------------------------------------------
    # Save best model + scaler
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"🏆 Best Model: {best_name} ({best_acc * 100:.2f}%)")
    print("=" * 60)

    joblib.dump(best_model, f"{MODEL_DIR}/placement_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(used_features, f"{MODEL_DIR}/features.pkl")

    print(f"\n✅ Saved:")
    print(f"   • {MODEL_DIR}/placement_model.pkl")
    print(f"   • {MODEL_DIR}/scaler.pkl")
    print(f"   • {MODEL_DIR}/features.pkl")

    print(f"\n📊 Classification Report ({best_name}):")
    print(classification_report(y_test, best_model.predict(X_test),
                                target_names=["Not Placed", "Placed"]))

    return best_name, best_acc


if __name__ == "__main__":
    train()