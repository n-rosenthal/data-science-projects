# =========================================================
# SPACESHIP TITANIC
# FAST HIGH-PERFORMANCE ENSEMBLE
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42

N_SPLITS = 3

TARGET = "Transported"

ID_COL = "PassengerId"

# =========================================================
# LOAD DATA
# =========================================================

TRAIN_PATH = "data/processed/train_processed.csv"

TEST_PATH  = "data/processed/test_processed.csv"

train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)

print("\n==============================")
print("DATA LOADED")
print("==============================")

print("Train Shape:", train_df.shape)

print("Test Shape :", test_df.shape)

# =========================================================
# STORE TEST IDS
# =========================================================

test_ids = test_df[ID_COL].copy()

# =========================================================
# TARGET
# =========================================================

y = train_df[TARGET].astype(int)

# =========================================================
# FEATURES
# =========================================================

X = train_df.drop(columns=[TARGET])

# =========================================================
# REMOVE ID
# =========================================================

X = X.drop(columns=[ID_COL])

test_df = test_df.drop(columns=[ID_COL])

# =========================================================
# REMOVE TARGET FROM TEST IF EXISTS
# =========================================================

if TARGET in test_df.columns:

    test_df = test_df.drop(columns=[TARGET])

# =========================================================
# ALIGN TRAIN / TEST
# =========================================================

missing_in_test = set(X.columns) - set(test_df.columns)

missing_in_train = set(test_df.columns) - set(X.columns)

for col in missing_in_test:

    test_df[col] = np.nan

for col in missing_in_train:

    X[col] = np.nan

test_df = test_df[X.columns]

# =========================================================
# CONCAT
# =========================================================

combined = pd.concat(
    [X, test_df],
    axis=0,
    ignore_index=True
)

print("\nCombined Shape:", combined.shape)

# =========================================================
# BOOLEAN -> INT
# =========================================================

bool_cols = combined.select_dtypes(
    include=["bool"]
).columns

for col in bool_cols:

    combined[col] = combined[col].astype(int)

# =========================================================
# NUMERICAL IMPUTATION
# =========================================================

num_cols = combined.select_dtypes(
    include=[np.number]
).columns

for col in num_cols:

    combined[col] = combined[col].fillna(
        combined[col].median()
    )

# =========================================================
# CATEGORICAL IMPUTATION
# =========================================================

cat_cols = combined.select_dtypes(
    include=["object", "category"]
).columns

for col in cat_cols:

    combined[col] = combined[col].fillna(
        "Unknown"
    )

# =========================================================
# ENCODING
# =========================================================

if len(cat_cols) > 0:

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    combined[cat_cols] = encoder.fit_transform(
        combined[cat_cols].astype(str)
    )

# =========================================================
# FINAL CHECK
# =========================================================

final_missing = combined.isnull().sum().sum()

print("\nFinal Missing Values:", final_missing)

assert final_missing == 0

# =========================================================
# SPLIT AGAIN
# =========================================================

X = combined.iloc[:len(train_df)].copy()

test_df = combined.iloc[len(train_df):].copy()

print("\nFinal Train Shape:", X.shape)

print("Final Test Shape :", test_df.shape)

# =========================================================
# CROSS VALIDATION
# =========================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# =========================================================
# MODELS
# =========================================================

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    verbose=0,
    random_seed=RANDOM_STATE
)

lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    random_state=RANDOM_STATE
)

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    eval_metric="logloss",
    random_state=RANDOM_STATE
)

et_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=14,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=14,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# =========================================================
# STACKING ENSEMBLE
# =========================================================

improved_model = StackingClassifier(

    estimators=[

        ("cat", cat_model),

        ("lgbm", lgbm_model),

        ("xgb", xgb_model),

        ("et", et_model),

        ("rf", rf_model)

    ],

    final_estimator=LogisticRegression(
        max_iter=5000
    ),

    stack_method="predict_proba",

    cv=5,

    n_jobs=-1
)

# =========================================================
# OOF PREDICTIONS
# =========================================================

print("\n==============================")
print("GENERATING OOF PREDICTIONS")
print("==============================")

oof_probs = cross_val_predict(
    improved_model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

# =========================================================
# THRESHOLD OPTIMIZATION
# =========================================================

print("\n==============================")
print("OPTIMIZING THRESHOLD")
print("==============================")

best_threshold = 0.5

best_acc = 0

for threshold in np.arange(0.30, 0.71, 0.005):

    preds = (
        oof_probs > threshold
    ).astype(int)

    acc = accuracy_score(
        y,
        preds
    )

    if acc > best_acc:

        best_acc = acc

        best_threshold = threshold

# =========================================================
# FINAL METRICS
# =========================================================

final_preds = (
    oof_probs > best_threshold
).astype(int)

final_acc = accuracy_score(
    y,
    final_preds
)

final_f1 = f1_score(
    y,
    final_preds
)

final_auc = roc_auc_score(
    y,
    oof_probs
)

print("\n==============================")
print("FINAL METRICS")
print("==============================")

print(f"Best Threshold : {best_threshold:.4f}")

print(f"OOF Accuracy   : {final_acc:.6f}")

print(f"OOF F1 Score   : {final_f1:.6f}")

print(f"OOF ROC AUC    : {final_auc:.6f}")

# =========================================================
# TRAIN FINAL MODEL
# =========================================================

print("\n==============================")
print("TRAINING FINAL MODEL")
print("==============================")

final_model = improved_model

final_model.fit(X, y)

# =========================================================
# TEST PREDICTIONS
# =========================================================

test_probabilities = final_model.predict_proba(
    test_df
)[:, 1]

test_predictions = (
    test_probabilities > best_threshold
).astype(bool)

# =========================================================
# SUBMISSION
# =========================================================

submission = pd.DataFrame({

    ID_COL: test_ids,

    "prediction": test_predictions

})

print("\n==============================")
print("SUBMISSION HEAD")
print("==============================")

print(submission.head())

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nsubmission.csv generated successfully.")