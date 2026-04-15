import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Carregar dados
dados = load_iris()
X = dados.data
y = dados.target

# 2. Normalizar (OBRIGATÓRIO para KNN)
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# 3. Dividir treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_normalizado, y, test_size=0.2, random_state=42
)

# 4. Testar vários valores de K com validação cruzada
valores_k = [1, 3, 5, 7, 9, 11]
medias = []

for k in valores_k:
    modelo = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(modelo, X_treino, y_treino, cv=10)
    medias.append(scores.mean())
    print(f"K={k} -> Exactidão média: {scores.mean():.4f}")

# 5. Gráfico
plt.plot(valores_k, medias, marker='o')
plt.title("Exactidão média vs K")
plt.xlabel("Valor de K")
plt.ylabel("Exactidão média")
plt.grid()
plt.show()

# 6. Escolher melhor K
melhor_k = valores_k[np.argmax(medias)]
print(f"\nMelhor K: {melhor_k}")


# 7. Treinar modelo final
modelo_final = KNeighborsClassifier(n_neighbors=melhor_k)
modelo_final.fit(X_treino, y_treino)


# 8. Avaliar no teste
previsoes = modelo_final.predict(X_teste)
print("Exactidão no teste:", accuracy_score(y_teste, previsoes))