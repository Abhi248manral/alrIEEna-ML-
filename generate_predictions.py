import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
import lightgbm as lgb
from xgboost import XGBClassifier

# ==============================
# Load Data
# ==============================

print("Loading data...")
train = pd.read_csv("TRAIN.csv")
test = pd.read_csv("TEST.csv")

y = train["Class"]
X = train.drop(columns=["Class"])

test_ids = test["ID"]
X_test = test.drop(columns=["ID"])

feature_cols = [c for c in X.columns if c.startswith("F")]

# ==============================
# Remove Duplicates
# ==============================

combined = pd.concat([X, y], axis=1)
combined = combined.drop_duplicates()
X = combined.drop(columns=["Class"]).reset_index(drop=True)
y = combined["Class"].reset_index(drop=True)
print(f"Train: {X.shape[0]} rows after dedup")

# ==============================
# Feature Engineering
# ==============================

def create_features(df):
    df = df.copy()
    fc = [c for c in df.columns if c.startswith("F")]
    fd = df[fc]

    # Row-wise stats
    df["r_mean"] = fd.mean(axis=1)
    df["r_std"] = fd.std(axis=1)
    df["r_max"] = fd.max(axis=1)
    df["r_min"] = fd.min(axis=1)
    df["r_range"] = df["r_max"] - df["r_min"]
    df["r_median"] = fd.median(axis=1)
    df["r_sum"] = fd.sum(axis=1)
    df["r_skew"] = fd.skew(axis=1)
    df["r_kurt"] = fd.kurtosis(axis=1)

    # Percentiles
    df["r_q10"] = fd.quantile(0.10, axis=1)
    df["r_q25"] = fd.quantile(0.25, axis=1)
    df["r_q75"] = fd.quantile(0.75, axis=1)
    df["r_q90"] = fd.quantile(0.90, axis=1)
    df["r_iqr"] = df["r_q75"] - df["r_q25"]

    # Counts
    df["r_zeros"] = (fd == 0).sum(axis=1)
    df["r_pos"] = (fd > 0).sum(axis=1)
    df["r_neg"] = (fd < 0).sum(axis=1)

    # Ratios
    df["r_cv"] = df["r_std"] / (df["r_mean"].abs() + 1e-8)
    df["r_mean2std"] = df["r_mean"] / (df["r_std"] + 1e-8)

    # Energy
    df["r_energy"] = (fd ** 2).sum(axis=1)
    df["r_l2"] = np.sqrt(df["r_energy"])
    df["r_abs_mean"] = fd.abs().mean(axis=1)

    # Top pairwise interactions (first 10 features)
    top = fc[:10]
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            df[f"d_{top[i]}_{top[j]}"] = df[top[i]] - df[top[j]]
            df[f"m_{top[i]}_{top[j]}"] = df[top[i]] * df[top[j]]

    return df

print("Engineering features...")
X = create_features(X)
X_test = create_features(X_test)
X = X.fillna(0)
X_test = X_test.fillna(0)
print(f"Features: {X.shape[1]}")

# ==============================
# 5-Fold CV with 5 Models
# ==============================

N_SPLITS = 5
folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def train_lgb(X, y, X_test, folds):
    oof = np.zeros(len(X))
    test_p = np.zeros(len(X_test))
    for fold, (tr, va) in enumerate(folds.split(X, y)):
        print(f"  Fold {fold+1}", end=" ", flush=True)
        m = lgb.LGBMClassifier(
            n_estimators=3000, learning_rate=0.01, num_leaves=127,
            max_depth=-1, feature_fraction=0.7, bagging_fraction=0.7,
            bagging_freq=5, min_child_samples=10, reg_alpha=0.05,
            reg_lambda=0.1, verbose=-1, random_state=42+fold,
        )
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test_p += m.predict_proba(X_test)[:, 1] / N_SPLITS
    return oof, test_p

def train_xgb(X, y, X_test, folds):
    oof = np.zeros(len(X))
    test_p = np.zeros(len(X_test))
    for fold, (tr, va) in enumerate(folds.split(X, y)):
        print(f"  Fold {fold+1}", end=" ", flush=True)
        m = XGBClassifier(
            n_estimators=3000, learning_rate=0.01, max_depth=10,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
            reg_alpha=0.05, reg_lambda=1.0, eval_metric="logloss",
            verbosity=0, random_state=42+fold, tree_method="hist",
            early_stopping_rounds=150,
        )
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test_p += m.predict_proba(X_test)[:, 1] / N_SPLITS
    return oof, test_p

def train_hgb(X, y, X_test, folds):
    oof = np.zeros(len(X))
    test_p = np.zeros(len(X_test))
    for fold, (tr, va) in enumerate(folds.split(X, y)):
        print(f"  Fold {fold+1}", end=" ", flush=True)
        m = HistGradientBoostingClassifier(
            max_iter=2000, learning_rate=0.01, max_depth=10,
            max_leaf_nodes=127, min_samples_leaf=10,
            l2_regularization=0.05, random_state=42+fold,
            early_stopping=True, n_iter_no_change=100, validation_fraction=0.1,
        )
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test_p += m.predict_proba(X_test)[:, 1] / N_SPLITS
    return oof, test_p

def train_et(X, y, X_test, folds):
    oof = np.zeros(len(X))
    test_p = np.zeros(len(X_test))
    for fold, (tr, va) in enumerate(folds.split(X, y)):
        print(f"  Fold {fold+1}", end=" ", flush=True)
        m = ExtraTreesClassifier(
            n_estimators=1500, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features=0.7, random_state=42+fold, n_jobs=-1,
        )
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test_p += m.predict_proba(X_test)[:, 1] / N_SPLITS
    return oof, test_p

def train_rf(X, y, X_test, folds):
    oof = np.zeros(len(X))
    test_p = np.zeros(len(X_test))
    for fold, (tr, va) in enumerate(folds.split(X, y)):
        print(f"  Fold {fold+1}", end=" ", flush=True)
        m = RandomForestClassifier(
            n_estimators=1500, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features=0.7, random_state=42+fold, n_jobs=-1,
        )
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test_p += m.predict_proba(X_test)[:, 1] / N_SPLITS
    return oof, test_p

# Train all models
print("\n=== LightGBM ===")
lgb_oof, lgb_test = train_lgb(X, y, X_test, folds)
lgb_acc = accuracy_score(y, (lgb_oof > 0.5).astype(int))
print(f"\n  OOF Accuracy: {lgb_acc:.6f}")

print("\n=== XGBoost ===")
xgb_oof, xgb_test = train_xgb(X, y, X_test, folds)
xgb_acc = accuracy_score(y, (xgb_oof > 0.5).astype(int))
print(f"\n  OOF Accuracy: {xgb_acc:.6f}")

print("\n=== HistGBM ===")
hgb_oof, hgb_test = train_hgb(X, y, X_test, folds)
hgb_acc = accuracy_score(y, (hgb_oof > 0.5).astype(int))
print(f"\n  OOF Accuracy: {hgb_acc:.6f}")

print("\n=== ExtraTrees ===")
et_oof, et_test = train_et(X, y, X_test, folds)
et_acc = accuracy_score(y, (et_oof > 0.5).astype(int))
print(f"\n  OOF Accuracy: {et_acc:.6f}")

print("\n=== RandomForest ===")
rf_oof, rf_test = train_rf(X, y, X_test, folds)
rf_acc = accuracy_score(y, (rf_oof > 0.5).astype(int))
print(f"\n  OOF Accuracy: {rf_acc:.6f}")

# ==============================
# Weighted Blend
# ==============================

print("\n=== Ensemble ===")
accs = np.array([lgb_acc, xgb_acc, hgb_acc, et_acc, rf_acc])
w = accs ** 5
w = w / w.sum()
print(f"Weights: LGB={w[0]:.3f} XGB={w[1]:.3f} HGB={w[2]:.3f} ET={w[3]:.3f} RF={w[4]:.3f}")

oof_blend = w[0]*lgb_oof + w[1]*xgb_oof + w[2]*hgb_oof + w[3]*et_oof + w[4]*rf_oof
test_blend = w[0]*lgb_test + w[1]*xgb_test + w[2]*hgb_test + w[3]*et_test + w[4]*rf_test

# Threshold optimization
best_score, best_thresh = 0, 0.5
for t in np.arange(0.20, 0.80, 0.001):
    s = accuracy_score(y, (oof_blend > t).astype(int))
    if s > best_score:
        best_score, best_thresh = s, t

print(f"Ensemble OOF Accuracy: {best_score:.6f} (thresh={best_thresh:.4f})")

# ==============================
# Pseudo-Labeling (3 rounds)
# ==============================

print("\n=== Pseudo-Labeling ===")
current_blend = test_blend.copy()

for rnd in range(3):
    confident = (current_blend < 0.03) | (current_blend > 0.97)
    n_conf = confident.sum()
    print(f"Round {rnd+1}: {n_conf} confident samples")
    if n_conf < 100:
        break

    pl = (current_blend[confident] > 0.5).astype(int)
    Xa = pd.concat([X, X_test[confident].copy()], axis=0).reset_index(drop=True)
    ya = pd.concat([y, pd.Series(pl.values)], axis=0).reset_index(drop=True)

    new_preds = np.zeros(len(X_test))
    n = 0

    for seed in [42, 123, 456]:
        m = lgb.LGBMClassifier(
            n_estimators=3000, learning_rate=0.01, num_leaves=127,
            feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5,
            min_child_samples=10, verbose=-1, random_state=seed)
        m.fit(Xa, ya)
        new_preds += m.predict_proba(X_test)[:, 1]; n += 1

    for seed in [42, 123, 456]:
        m = XGBClassifier(
            n_estimators=3000, learning_rate=0.01, max_depth=10,
            subsample=0.7, colsample_bytree=0.7, verbosity=0,
            random_state=seed, tree_method="hist")
        m.fit(Xa, ya)
        new_preds += m.predict_proba(X_test)[:, 1]; n += 1

    for seed in [42, 123, 456]:
        m = HistGradientBoostingClassifier(
            max_iter=2000, learning_rate=0.01, max_depth=10,
            max_leaf_nodes=127, random_state=seed)
        m.fit(Xa, ya)
        new_preds += m.predict_proba(X_test)[:, 1]; n += 1

    new_preds /= n
    current_blend = 0.4 * current_blend + 0.6 * new_preds

test_blend = current_blend

# ==============================
# Majority Voting Override
# ==============================

all_tests = [lgb_test, xgb_test, hgb_test, et_test, rf_test]
votes = np.zeros(len(X_test))
for t in all_tests:
    votes += (t > 0.5).astype(int)
vote_pct = votes / len(all_tests)

final_preds = (test_blend > best_thresh).astype(int)

# Override when strong consensus
final_preds[vote_pct <= 0.0] = 0
final_preds[vote_pct >= 1.0] = 1

# ==============================
# Output
# ==============================

submission = pd.DataFrame({"ID": test_ids, "CLASS": final_preds})
submission.to_csv("FINAL.csv", index=False)

print(f"\nFINAL.csv: {len(submission)} rows")
print(f"Distribution: 0={(final_preds==0).sum()}, 1={(final_preds==1).sum()}")
print(f"CV Accuracy: {best_score:.6f}")
print(f"Individual: LGB={lgb_acc:.4f} XGB={xgb_acc:.4f} HGB={hgb_acc:.4f} ET={et_acc:.4f} RF={rf_acc:.4f}")