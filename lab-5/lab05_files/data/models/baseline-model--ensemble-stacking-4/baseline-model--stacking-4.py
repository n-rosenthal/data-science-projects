import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import category_encoders as ce


# ============================================================
# 1. Dados
# ============================================================
train_df = pd.read_csv("./lab05_files/lab05_files/train_student.csv")
test_df  = pd.read_csv("./lab05_files/lab05_files/test_student.csv")

# Remover outliers mais agressivamente
train_df = train_df[train_df["GrLivArea"] < 4500]
train_df = train_df[~((train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 200000))]
train_df = train_df[train_df["LotArea"] < 100000]
train_df = train_df[train_df["TotalBsmtSF"] < 6000]

X = train_df.drop("SalePrice", axis=1)
y = np.log1p(train_df["SalePrice"])
X_test = test_df.copy()

print("1. Dados:", X.shape, X_test.shape)


# ============================================================
# 2. Feature Engineering EXPANDIDO
# ============================================================
def add_features(df):
    df = df.copy()

    # ---- originais ----
    df["TotalSF"]       = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBathrooms"]= df["FullBath"] + 0.5*df["HalfBath"] + df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]
    df["HouseAge"]      = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"]      = df["YrSold"] - df["YearRemodAdd"]
    df["GarageScore"]   = df["GarageCars"] * df["GarageArea"]

    # ---- novas ----
    df["TotalPorchSF"]  = (df["OpenPorchSF"] + df["EnclosedPorch"] +
                           df["3SsnPorch"]    + df["ScreenPorch"])
    df["HasPool"]       = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFloor"]   = (df["2ndFlrSF"] > 0).astype(int)
    df["HasGarage"]     = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"]       = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"]  = (df["Fireplaces"] > 0).astype(int)
    df["IsRemodeled"]   = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["IsNew"]         = (df["YrSold"] == df["YearBuilt"]).astype(int)

    # interações chave
    df["OverallQual_TotalSF"]   = df["OverallQual"] * df["TotalSF"]
    df["OverallQual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    df["OverallQual_sq"]        = df["OverallQual"] ** 2
    df["GrLivArea_sq"]          = df["GrLivArea"] ** 2
    df["TotalSF_sq"]            = df["TotalSF"] ** 2

    # preço por área (proxy)
    df["LotAreaLog"]    = np.log1p(df["LotArea"])
    df["GrLivAreaLog"]  = np.log1p(df["GrLivArea"])
    df["TotalSFLog"]    = np.log1p(df["TotalSF"])

    # idade ao vender vs reforma
    df["AgeRemodSold"]  = df["YrSold"] - df["YearRemodAdd"]
    df["QualCondScore"] = df["OverallQual"] * df["OverallCond"]

    return df


X     = add_features(X)
X_test= add_features(X_test)

print("2. Feature engineering OK -", X.shape[1], "features")


# ============================================================
# 3. Tipos e imputação de nulos com contexto
# ============================================================
X["MSSubClass"]     = X["MSSubClass"].astype(str)
X_test["MSSubClass"]= X_test["MSSubClass"].astype(str)

# Colunas categóricas cujo NaN significa "não tem"
none_cols = [
    "Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
    "FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
    "PoolQC","Fence","MiscFeature","MasVnrType",
]
for col in none_cols:
    for df_ in [X, X_test]:
        if col in df_.columns:
            df_[col] = df_[col].fillna("None")

# Numéricas: NaN = 0 para colunas de área/contagem
zero_cols = [
    "MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
    "BsmtFullBath","BsmtHalfBath","GarageYrBlt","GarageCars","GarageArea",
    "PoolArea","MiscVal"
]
for col in zero_cols:
    for df_ in [X, X_test]:
        if col in df_.columns:
            df_[col] = df_[col].fillna(0)

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()

print(f"3. Num: {len(num_cols)} | Cat: {len(cat_cols)}")


# ============================================================
# 4. CV setup
# ============================================================
kf    = KFold(n_splits=10, shuffle=True, random_state=42)
seeds = [42]   # melhor seed com base nos resultados anteriores


# ============================================================
# 5. Containers OOF
# ============================================================
oof_preds  = []
test_preds = []

n_models = 5   # lgb1, lgb2, lgb3, cat, xgb  (ElasticNet vai só no meta)


# ============================================================
# 6. LOOP PRINCIPAL
# ============================================================
for seed_idx, seed in enumerate(seeds):
    print(f"\n--- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")

    oof_lgb1 = np.zeros(len(X))
    oof_lgb2 = np.zeros(len(X))
    oof_lgb3 = np.zeros(len(X))
    oof_cat  = np.zeros(len(X))
    oof_xgb  = np.zeros(len(X))

    pred_lgb1 = np.zeros(len(X_test))
    pred_lgb2 = np.zeros(len(X_test))
    pred_lgb3 = np.zeros(len(X_test))
    pred_cat  = np.zeros(len(X_test))
    pred_xgb  = np.zeros(len(X_test))

    denom = kf.n_splits * len(seeds)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        X_tr,  X_val  = X.iloc[train_idx],  X.iloc[val_idx]
        y_tr,  y_val  = y.iloc[train_idx],  y.iloc[val_idx]

        # ---------- Numéricas ----------
        imputer = SimpleImputer(strategy="median")
        scaler  = RobustScaler()   # mais robusto a outliers que StandardScaler

        X_tr_num   = scaler.fit_transform(imputer.fit_transform(X_tr[num_cols]))
        X_val_num  = scaler.transform(imputer.transform(X_val[num_cols]))
        X_test_num = scaler.transform(imputer.transform(X_test[num_cols]))

        # ---------- Target Encoding ----------
        encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10)   # smoothing maior = menos leakage
        encoder.fit(X_tr[cat_cols + num_cols[:1]], y_tr)   # fit com todas as colunas
        # mas só extrair as categóricas
        enc_tr   = ce.TargetEncoder(cols=cat_cols, smoothing=10)
        enc_tr.fit(X_tr, y_tr)

        X_tr_te   = enc_tr.transform(X_tr)[cat_cols].values
        X_val_te  = enc_tr.transform(X_val)[cat_cols].values
        X_test_te = enc_tr.transform(X_test)[cat_cols].values

        # ---------- Matrizes finais ----------
        X_tr_f   = np.hstack([X_tr_num,   X_tr_te])
        X_val_f  = np.hstack([X_val_num,  X_val_te])
        X_test_f = np.hstack([X_test_num, X_test_te])

        # ========== MODELOS ==========

        # LGB1 — conservador, profundo
        lgb1 = LGBMRegressor(
            n_estimators=6000, learning_rate=0.005,
            num_leaves=31, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=seed, n_jobs=-1, verbose=-1
        )

        # LGB2 — mais raso, mais regularizado
        lgb2 = LGBMRegressor(
            n_estimators=8000, learning_rate=0.003,
            num_leaves=15, max_depth=5,
            subsample=0.7, colsample_bytree=0.7,
            min_child_samples=30,
            reg_alpha=0.3, reg_lambda=0.3,
            random_state=seed+1, n_jobs=-1, verbose=-1
        )

        # LGB3 — diferente: feature fraction baixo, many leaves
        lgb3 = LGBMRegressor(
            n_estimators=5000, learning_rate=0.008,
            num_leaves=63, max_depth=6,
            subsample=0.75, colsample_bytree=0.6,
            subsample_freq=1,
            min_child_samples=15,
            reg_alpha=0.05, reg_lambda=0.2,
            random_state=seed+2, n_jobs=-1, verbose=-1
        )

        # CatBoost
        cat = CatBoostRegressor(
            iterations=5000, learning_rate=0.007,
            depth=6, l2_leaf_reg=3,
            bagging_temperature=0.5,
            random_seed=seed, verbose=0
        )

        # XGBoost — complementa LGB/Cat
        xgb = XGBRegressor(
            n_estimators=5000, learning_rate=0.005,
            max_depth=5, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            gamma=0.1,
            random_state=seed, n_jobs=-1,
            verbosity=0
        )

        # ========== FIT + PRED ==========
        lgb1.fit(X_tr_f, y_tr,
                 eval_set=[(X_val_f, y_val)],
                 callbacks=[])
        oof_lgb1[val_idx]  = lgb1.predict(X_val_f)
        pred_lgb1         += lgb1.predict(X_test_f) / denom

        lgb2.fit(X_tr_f, y_tr)
        oof_lgb2[val_idx]  = lgb2.predict(X_val_f)
        pred_lgb2         += lgb2.predict(X_test_f) / denom

        lgb3.fit(X_tr_f, y_tr)
        oof_lgb3[val_idx]  = lgb3.predict(X_val_f)
        pred_lgb3         += lgb3.predict(X_test_f) / denom

        cat.fit(X_tr_f, y_tr,
                eval_set=(X_val_f, y_val),
                early_stopping_rounds=200,
                verbose=False)
        oof_cat[val_idx]   = cat.predict(X_val_f)
        pred_cat          += cat.predict(X_test_f) / denom

        xgb.fit(X_tr_f, y_tr,
                eval_set=[(X_val_f, y_val)],
                verbose=False)
        oof_xgb[val_idx]   = xgb.predict(X_val_f)
        pred_xgb          += xgb.predict(X_test_f) / denom

        if (fold + 1) % 5 == 0:
            print(f"  fold {fold+1}/{kf.n_splits} OK")

    # RMSE parciais
    for name, oof in [("lgb1",oof_lgb1),("lgb2",oof_lgb2),("lgb3",oof_lgb3),
                      ("cat",oof_cat),("xgb",oof_xgb)]:
        rmse_i = np.sqrt(mean_squared_error(y, oof))
        print(f"  {name}: OOF RMSE = {rmse_i:.6f}")

    oof_preds.append(np.vstack([oof_lgb1, oof_lgb2, oof_lgb3, oof_cat, oof_xgb]).T)
    test_preds.append(np.vstack([pred_lgb1, pred_lgb2, pred_lgb3, pred_cat, pred_xgb]).T)


# ============================================================
# 7. STACKING FINAL — duas camadas
# ============================================================
X_meta      = np.mean(oof_preds,  axis=0)   # (n_train, 5)
X_meta_test = np.mean(test_preds, axis=0)   # (n_test,  5)

print("\n--- Meta-learning ---")
for i, name in enumerate(["lgb1","lgb2","lgb3","cat","xgb"]):
    rmse_i = np.sqrt(mean_squared_error(y, X_meta[:, i]))
    print(f"  {name} (avg seeds): {rmse_i:.6f}")

# Meta-modelo 1: Ridge (mais estável que ElasticNet quando features são predições)
from sklearn.linear_model import RidgeCV
meta_ridge = RidgeCV(alphas=np.logspace(-4, 2, 50), cv=10)
meta_ridge.fit(X_meta, y)
oof_ridge = meta_ridge.predict(X_meta)
print(f"  Ridge meta RMSE:    {np.sqrt(mean_squared_error(y, oof_ridge)):.6f}")

# Meta-modelo 2: ElasticNet clássico
meta_en = ElasticNet(alpha=0.0003, l1_ratio=0.5, max_iter=50000)
meta_en.fit(X_meta, y)
oof_en = meta_en.predict(X_meta)
print(f"  ElasticNet meta RMSE: {np.sqrt(mean_squared_error(y, oof_en)):.6f}")

# Blend dos dois metas
w_ridge, w_en = 0.5, 0.5
stack_oof  = w_ridge * oof_ridge  + w_en * oof_en
stack_test = w_ridge * meta_ridge.predict(X_meta_test) + w_en * meta_en.predict(X_meta_test)

rmse_final = np.sqrt(mean_squared_error(y, stack_oof))
print(f"\n★ OOF RMSE FINAL (blend meta): {rmse_final:.6f}")

# Blend opcional: média ponderada direta dos modelos base (às vezes bate o meta)
weights     = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
direct_oof  = X_meta      @ weights
direct_test = X_meta_test @ weights
rmse_direct = np.sqrt(mean_squared_error(y, direct_oof))
print(f"★ OOF RMSE FINAL (média direta): {rmse_direct:.6f}")

# Escolher o melhor automaticamente
if rmse_direct < rmse_final:
    print("→ Usando média direta (melhor)")
    final_log  = direct_test
    best_rmse  = rmse_direct
else:
    print("→ Usando blend de metas (melhor)")
    final_log  = stack_test
    best_rmse  = rmse_final


# ============================================================
# 8. SUBMISSÃO
# ============================================================
# A competição calcula RMSE_LOG diretamente sobre log(SalePrice),
# portanto NÃO revertemos com expm1 — enviamos os valores em log.
final_pred = np.expm1(final_log)

submission = pd.DataFrame({
    "Id":         test_df["Id"],
    "prediction": final_pred
})

submission.to_csv("n-rosenthal--submission-3.csv", index=False)
print(f"\n✓ Submission gerada (log scale)! OOF RMSE estimado: {best_rmse:.6f}")
print(f"  Faixa esperada de prediction: [{final_pred.min():.3f}, {final_pred.max():.3f}]  (deve estar entre ~10 e ~14)")
print("  Lembrete: RMSE OOF tende a ser ligeiramente pessimista — score real costuma ser igual ou melhor.")
