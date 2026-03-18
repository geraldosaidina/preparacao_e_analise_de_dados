import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.impute import SimpleImputer, KNNImputer

# =========================
# 1. Ler dataset
# =========================
with ZipFile("pacientes.csv") as z:
    with z.open("cardio_data_processed.csv") as f:
        df = pd.read_csv(f)

print("Colunas originais:")
print(df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())

# =========================
# 2. Mapear colunas para português
# =========================
df = df.rename(columns={
    "age_years": "idade",
    "weight": "peso",
    "gluc": "glicemia",
    "cardio": "diagnostico"
})

# Criar coluna 'pressao' a partir da média entre sistólica e diastólica
df["pressao"] = (df["ap_hi"] + df["ap_lo"]) / 2

# Ficar apenas com as colunas necessárias
df = df[["idade", "peso", "pressao", "glicemia", "diagnostico"]].copy()

print("\nDados após mapeamento:")
print(df.head())

# =========================
# 3. Introduzir problemas para reproduzir o cenário
# =========================
np.random.seed(42)

# 15% de glicemia em falta
idx_glicemia = df.sample(frac=0.15, random_state=42).index
df.loc[idx_glicemia, "glicemia"] = np.nan

# 8% de pressão em falta
idx_pressao = df.sample(frac=0.08, random_state=24).index
df.loc[idx_pressao, "pressao"] = np.nan

# 3 idades negativas
idx_idades_negativas = df.sample(n=3, random_state=7).index
df.loc[idx_idades_negativas, "idade"] = [-5, -12, -1]

print("\nValores em falta antes do tratamento:")
print(df.isnull().sum())

print("\nIdades negativas antes do tratamento:")
print(df[df["idade"] < 0])

# =========================
# 4. Corrigir idades negativas
# =========================
df.loc[df["idade"] < 0, "idade"] = np.nan

# =========================
# 5. Imputação com mediana
# =========================
imputer_mediana = SimpleImputer(strategy="median")
df_mediana = df.copy()
df_mediana[["idade", "peso", "pressao", "glicemia"]] = imputer_mediana.fit_transform(
    df_mediana[["idade", "peso", "pressao", "glicemia"]]
)

print("\nValores em falta após imputação com mediana:")
print(df_mediana.isnull().sum())

print("\nDados tratados com mediana:")
print(df_mediana.head())

# =========================
# 6. Imputação com KNN
# =========================
imputer_knn = KNNImputer(n_neighbors=5)
df_knn = df.copy()
df_knn[["idade", "peso", "pressao", "glicemia"]] = imputer_knn.fit_transform(
    df_knn[["idade", "peso", "pressao", "glicemia"]]
)

print("\nValores em falta após imputação com KNN:")
print(df_knn.isnull().sum())

print("\nDados tratados com KNN:")
print(df_knn.head())