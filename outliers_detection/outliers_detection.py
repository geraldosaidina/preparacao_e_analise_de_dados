import pandas as pd
import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt

# =========================
# 1. Ler dataset
# =========================
with ZipFile("transaccoes.csv") as z:
    with z.open("Synthetic_Financial_datasets_log.csv") as f:
        df = pd.read_csv(f)

print("Colunas originais:")
print(df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())

# =========================
# 2. Mapear colunas para português
# =========================
df = df.rename(columns={
    "amount": "valor",
    "type": "tipo",
    "oldbalanceOrg": "saldo",
    "step": "hora"
})

df = df[["valor", "tipo", "saldo", "hora", "isFraud"]].copy()

print("\nDados após mapeamento:")
print(df.head())

# =========================
# 3. Introduzir problemas para reproduzir o cenário
# =========================
np.random.seed(42)

# Inserir alguns valores negativos impossíveis
idx_negativos = df.sample(n=5, random_state=42).index
df.loc[idx_negativos, "valor"] = -df.loc[idx_negativos, "valor"].abs()

# Inserir alguns outliers extremos
idx_outliers = df.sample(n=10, random_state=10).index
df.loc[idx_outliers, "valor"] = df["valor"].median() * 50

print("\nResumo estatístico antes do tratamento:")
print(df["valor"].describe())

# =========================
# 4. Remover valores negativos
# =========================
df = df[df["valor"] >= 0].copy()

# =========================
# 5. Detectar outliers pelo IQR
# =========================
Q1 = df["valor"].quantile(0.25)
Q3 = df["valor"].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers_iqr = df[(df["valor"] < limite_inferior) | (df["valor"] > limite_superior)]
print(f"\nNúmero de outliers detectados com IQR: {len(outliers_iqr)}")

# =========================
# 6. Detectar outliers por Z-score
# =========================
media = df["valor"].mean()
desvio = df["valor"].std()

df["z_score"] = (df["valor"] - media) / desvio
outliers_z = df[df["z_score"].abs() > 3]
print(f"Número de outliers detectados com Z-score: {len(outliers_z)}")

# =========================
# 7. Tratar com winsorizing usando IQR
# =========================
df_tratado = df.copy()
df_tratado["valor"] = np.where(df_tratado["valor"] > limite_superior, limite_superior, df_tratado["valor"])
df_tratado["valor"] = np.where(df_tratado["valor"] < limite_inferior, limite_inferior, df_tratado["valor"])

print("\nResumo estatístico após tratamento:")
print(df_tratado["valor"].describe())

# =========================
# 8. Visualização
# =========================
plt.figure(figsize=(10, 5))
plt.boxplot(df["valor"], vert=False)
plt.title("Boxplot antes do tratamento")
plt.show()

plt.figure(figsize=(10, 5))
plt.boxplot(df_tratado["valor"], vert=False)
plt.title("Boxplot após tratamento")
plt.show()