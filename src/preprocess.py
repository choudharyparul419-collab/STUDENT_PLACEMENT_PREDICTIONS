"""
Data preprocessing for Student Placement Prediction.
Uses the REAL dataset: student_placement_prediction_dataset_2026.csv
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DATASET_PATH = "student_placement_prediction_dataset_2026.csv"

# Features we will use for ML training (only numeric + useful)
FEATURES = [
    "cgpa",
    "aptitude_score",
    "communication_skill_score",
    "coding_skill_score",
    "logical_reasoning_score",
    "internships_count",
    "projects_count",
    "certifications_count",
    "backlogs",
    "attendance_percentage",
    "mock_interview_score",
    "hackathons_participated",
    "github_repos",
    "linkedin_connections",
    "extracurricular_score",
    "leadership_score",
    "study_hours_per_day",
]

TARGET = "placement_status"   # values: "Placed" / "Not Placed"


def load_dataframe(path=DATASET_PATH):
    """Load raw CSV into pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Dataset not found at '{path}'.\n"
            f"   Please place the CSV file in the project ROOT folder."
        )
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def clean_dataframe(df):
    """Clean missing values and convert target to binary."""
    # Keep only features + target that exist in the CSV
    needed = [c for c in FEATURES if c in df.columns] + [TARGET]
    df = df[needed].copy()

    # Fill any missing numeric values with median
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET])

    # Convert target string -> 0/1
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    df[TARGET] = df[TARGET].map({"placed": 1, "not placed": 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    print(f"✅ Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Placed: {int((df[TARGET]==1).sum())} | "
          f"Not Placed: {int((df[TARGET]==0).sum())}")
    return df


def load_and_split(path=DATASET_PATH, test_size=0.2):
    """Returns scaled train/test sets + scaler."""
    df = load_dataframe(path)
    df = clean_dataframe(df)

    used_features = [c for c in FEATURES if c in df.columns]
    X = df[used_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, used_features


if __name__ == "__main__":
    load_and_split()