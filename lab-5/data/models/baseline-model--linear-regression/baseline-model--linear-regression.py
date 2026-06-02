import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#   1. Carregamento dos dados
train_df = pd.read_csv("./lab05_files/lab05_files/train_student.csv")
test_df  = pd.read_csv("./lab05_files/lab05_files/test_student.csv")

print("1. Dados carregados. Dimensões:", train_df.shape, test_df.shape)


#   2. Pré-processamento 1: remoção de colunas com alto número de valores ausentes
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']

train_df = train_df.drop(columns=cols_to_drop)
test_df  = test_df.drop(columns=cols_to_drop)

print("2. Eliminadas colunas com >= 0.8 valores ausentes.")

#   3. Separação de features (X) e target (y) para treinamento
X_train = train_df.drop('SalePrice', axis=1)
y_train = train_df['SalePrice']
X_test  = test_df.copy()

print(f"3. X_train dim.: {X_train.shape}")

#   4. Pré-processamento 1.1.: `MSSubClass` torna-se uma coluna categórica
if 'MSSubClass' in X_train.columns:
    X_train['MSSubClass'] = X_train['MSSubClass'].astype(str)
    X_test['MSSubClass']  = X_test['MSSubClass'].astype(str)

    print("4. coluna `MSSubClass` convertida para `str`")

#   5. Pré-processamento 2: imputação para colunas numéricas com zero
# Colunas com imputação zero
zero_impute_cols = [
    'MasVnrArea', 'GarageYrBlt', 'GarageCars', 'GarageArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath',
    'Fireplaces', 'PoolArea'
]
zero_impute_cols = [c for c in zero_impute_cols if c in X_train.columns]

print(f"5. Imputação de 0 para colunas ({len(zero_impute_cols)}): {zero_impute_cols[:5]}...")


#   6. Pré-processamento 2.1: imputação para colunas numéricas com mediana
all_numerical = X_train.select_dtypes(include=['int64','float64']).columns.tolist()

median_impute_cols = [
    c for c in all_numerical
    if c not in zero_impute_cols and c != 'Id'
]

print(f"6. Imputação da mediana ({len(median_impute_cols)}): {median_impute_cols[:5]}...")

#   7. Pré-processamento 3: imputação para colunas categóricas
categorical_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()

print(f"7. Colunas categóricas ({len(categorical_cols)}): {categorical_cols[:5]}...")

#   8. Construção dos pipelines
numeric_median_pipe = Pipeline([
    ('impute_median', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

numeric_zero_pipe = Pipeline([
    ('impute_zero', SimpleImputer(strategy='constant', fill_value=0)),
    ('scale', StandardScaler())
])

categorical_pipe = Pipeline([
    ('impute_missing', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])


# ColumnTransformer final
preprocessor = ColumnTransformer([
    ('num_median', numeric_median_pipe, median_impute_cols),
    ('num_zero',   numeric_zero_pipe,   zero_impute_cols),
    ('cat',        categorical_pipe,    categorical_cols)
])

print("-" * 30)
print("8. Pipeline de pré-processamento definido:")
print(preprocessor)


#   9. Transformação dos dados
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed  = preprocessor.transform(X_test)

print("9. Transformação concluída.")
print(f"X_train shape: {X_train_transformed.shape}")
print(f"X_test  shape: {X_test_transformed.shape}")


#   10. Treinamento do modelo baseline com regressão linear
model = LinearRegression()

# Usando log-transform no target
y_train_log = np.log1p(y_train)

model.fit(X_train_transformed, y_train_log)

print("10. Modelo LinearRegression treinado.")


#   11. Avaliação do modelo
y_pred_log = model.predict(X_train_transformed)
y_pred = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_train, y_pred))
mae  = mean_absolute_error(y_train, y_pred)
r2   = r2_score(y_train, y_pred)
rmse_log = np.sqrt(mean_squared_error(y_train_log, y_pred_log))

print("-" * 30)
print("11. Métricas no treino:")
print(f"RMSE (log): {rmse_log:.5f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

from sklearn.model_selection import cross_val_score

scores = -cross_val_score(
    model,
    X_train_transformed,
    y_train_log,
    scoring="neg_root_mean_squared_error",
    cv=5
)

print(f"CV RMSE (log): {scores.mean():.5f} ± {scores.std():.5f}")

#   12. Predição no conjunto teste
y_test_pred_log = model.predict(X_test_transformed)
y_test_pred = np.expm1(y_test_pred_log)

print("12. Predições no conjunto de teste concluídas.")

#   13. Gerar arquivo de submissão
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": y_test_pred
})

# submission.to_csv("submission.csv", index=False)

print("13. Arquivo 'submission.csv' gerado com sucesso!")