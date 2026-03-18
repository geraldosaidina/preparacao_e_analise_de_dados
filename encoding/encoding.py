import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# =========================
# 1. Ler dataset
# =========================
df = pd.read_csv("clientes_telecom.csv")

print("Colunas originais:")
print(df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())

# =========================
# 2. Mapear colunas para português
# =========================
df = df.rename(columns={
    "State": "regiao",
    "Contract": "contrato",
    "Internet Service": "plano",
    "Churn Label": "churn"
})

df = df[["regiao", "contrato", "plano", "churn"]].copy()

# =========================
# 3. Traduzir valores para português
# =========================
df["contrato"] = df["contrato"].replace({
    "Month-to-month": "mensal",
    "One year": "semestral",
    "Two year": "anual"
})

df["plano"] = df["plano"].replace({
    "DSL": "Basico",
    "No": "Standard",
    "Fiber optic": "Premium"
})

df["churn"] = df["churn"].replace({
    "Yes": "Sim",
    "No": "Nao"
})

print("\nDados após tradução:")
print(df.head())

# =========================
# 4. One-Hot Encoding para regiao
# =========================
encoder_regiao = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
regiao_encoded = encoder_regiao.fit_transform(df[["regiao"]])

df_regiao = pd.DataFrame(
    regiao_encoded,
    columns=encoder_regiao.get_feature_names_out(["regiao"])
)

# =========================
# 5. Label/Ordinal Encoding para contrato e plano
# =========================
encoder_ordinal = OrdinalEncoder(categories=[["mensal", "semestral", "anual"], ["Basico", "Standard", "Premium"]])
df[["contrato_encoded", "plano_encoded"]] = encoder_ordinal.fit_transform(df[["contrato", "plano"]])

# =========================
# 6. Churn binário
# =========================
df["churn_encoded"] = df["churn"].map({"Nao": 0, "Sim": 1})

# =========================
# 7. Resultado final
# =========================
df_final = pd.concat([df, df_regiao], axis=1)

print("\nDados codificados:")
print(df_final.head())