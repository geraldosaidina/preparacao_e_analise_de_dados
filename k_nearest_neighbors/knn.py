import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================
# 1. Ler dataset
# =========================
df = pd.read_csv("credit_risk_dataset.csv")

print("Colunas originais:")
print(df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())

# =========================
# 2. Mapear colunas para português
# =========================
df = df.rename(columns={
    "person_age": "idade",
    "person_income": "salario",
    "loan_amnt": "divida",
    "loan_status": "risco"
})

df = df[["idade", "salario", "divida", "risco"]].copy()

print("\nDados após mapeamento:")
print(df.head())

# =========================
# 3. Definir X e y
# =========================
X = df[["idade", "salario", "divida"]]
y = df["risco"]

# =========================
# 4. Divisão treino/teste
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5. KNN sem normalização
# =========================
knn_sem = KNeighborsClassifier(n_neighbors=5)
knn_sem.fit(X_train, y_train)
y_pred_sem = knn_sem.predict(X_test)
acc_sem = accuracy_score(y_test, y_pred_sem)

print(f"\nAcurácia sem normalização: {acc_sem:.4f}")

# =========================
# 6. KNN com MinMaxScaler
# =========================
minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train)
X_test_minmax = minmax.transform(X_test)

knn_minmax = KNeighborsClassifier(n_neighbors=5)
knn_minmax.fit(X_train_minmax, y_train)
y_pred_minmax = knn_minmax.predict(X_test_minmax)
acc_minmax = accuracy_score(y_test, y_pred_minmax)

print(f"Acurácia com MinMaxScaler: {acc_minmax:.4f}")

# =========================
# 7. KNN com StandardScaler
# =========================
standard = StandardScaler()
X_train_std = standard.fit_transform(X_train)
X_test_std = standard.transform(X_test)

knn_std = KNeighborsClassifier(n_neighbors=5)
knn_std.fit(X_train_std, y_train)
y_pred_std = knn_std.predict(X_test_std)
acc_std = accuracy_score(y_test, y_pred_std)

print(f"Acurácia com StandardScaler: {acc_std:.4f}")