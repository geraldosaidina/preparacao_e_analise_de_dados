import pandas as pd
import numpy as np

# =========================
# 1. Ler dataset
# =========================
df = pd.read_csv("vendas.csv")

print("Colunas originais:")
print(df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())

# =========================
# 2. Mapear colunas para português
# =========================
df = df.rename(columns={
    "Date": "data",
    "Weekly_Sales": "vendas"
})

df = df[["data", "vendas"]].copy()

# =========================
# 3. Converter data
# =========================
df["data"] = pd.to_datetime(df["data"], format="%d-%m-%Y")

# =========================
# 4. Criar atributos temporais
# =========================
df["dia_semana"] = df["data"].dt.dayofweek
df["mes"] = df["data"].dt.month
df["trimestre"] = df["data"].dt.quarter
df["e_fim_semana"] = (df["dia_semana"] >= 5).astype(int)

# dias até natal
natal_ano = pd.to_datetime(df["data"].dt.year.astype(str) + "-12-25")
df["dias_ate_natal"] = (natal_ano - df["data"]).dt.days
df["dias_ate_natal"] = df["dias_ate_natal"].apply(lambda x: x if x >= 0 else np.nan)

# Exemplo simples de feriados
feriados = [
    "2010-01-01", "2010-02-03", "2010-04-07", "2010-05-01", "2010-06-25",
    "2011-01-01", "2011-02-03", "2011-04-07", "2011-05-01", "2011-06-25",
    "2012-01-01", "2012-02-03", "2012-04-07", "2012-05-01", "2012-06-25"
]
feriados = pd.to_datetime(feriados)
df["e_feriado_nacional"] = df["data"].isin(feriados).astype(int)

# Média móvel de 7 dias
df = df.sort_values("data")
df["vendas_media_7dias"] = df["vendas"].rolling(window=7, min_periods=1).mean()

print("\nDados com features criadas:")
print(df.head(15))