import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. Ler dataset
# =========================
df = pd.read_csv("credit_risk_dataset.csv")

print("Colunas originais:")
print(df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())

# =========================
# 2. Renomear colunas para português
# =========================
df = df.rename(columns={
    "person_age": "idade",
    "person_income": "rendimento",
    "person_home_ownership": "habitacao",
    "person_emp_length": "anos_emprego",
    "loan_intent": "finalidade_emprestimo",
    "loan_grade": "classe_emprestimo",
    "loan_amnt": "montante_emprestimo",
    "loan_int_rate": "taxa_juro",
    "loan_status": "estado_emprestimo",
    "loan_percent_income": "percentagem_rendimento",
    "cb_person_default_on_file": "historico_incumprimento",
    "cb_person_cred_hist_length": "historico_credito"
})

# =========================
# 3. Introduzir alguns missing values artificiais
# =========================
np.random.seed(42)
idx_num = df.sample(frac=0.02, random_state=42).index
idx_cat = df.sample(frac=0.02, random_state=24).index

df.loc[idx_num, "taxa_juro"] = np.nan
df.loc[idx_cat, "habitacao"] = np.nan

print("\nValores em falta antes do pipeline:")
print(df.isnull().sum())

# =========================
# 4. Definir X e y
# =========================
X = df.drop(columns=["estado_emprestimo"])
y = df["estado_emprestimo"]

# =========================
# 5. Separar colunas numéricas e categóricas
# =========================
num_cols = [
    "idade",
    "rendimento",
    "anos_emprego",
    "montante_emprestimo",
    "taxa_juro",
    "percentagem_rendimento",
    "historico_credito"
]

cat_cols = [
    "habitacao",
    "finalidade_emprestimo",
    "classe_emprestimo",
    "historico_incumprimento"
]

# =========================
# 6. Pipeline numérico
# =========================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# =========================
# 7. Pipeline categórico
# =========================
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# =========================
# 8. Pré-processador
# =========================
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# =========================
# 9. Pipeline completo
# =========================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# =========================
# 10. Divisão treino/teste
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 11. Treinar pipeline
# =========================
pipeline.fit(X_train, y_train)

# =========================
# 12. Avaliação com Cross-Validation
# =========================
scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")

print("\nScores F1 em cada fold:")
print(scores)

print(f"\nF1 médio: {scores.mean():.4f}")
print(f"Desvio-padrão: {scores.std():.4f}")