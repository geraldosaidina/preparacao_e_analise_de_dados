import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

csv_path = Path(__file__).resolve().parent / "credito_knn.csv"
df = pd.read_csv(csv_path)

X = df[['idade', 'salario', 'divida']]
y = df['risco']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ajustar APENAS no treino
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)   # so transform!

# Sem normalizacao
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print('Sem norm:', accuracy_score(y_test, knn.predict(X_test)))

# Com normalizacao
knn.fit(X_train_sc, y_train)
print('Com norm:', accuracy_score(y_test, knn.predict(X_test_sc)))