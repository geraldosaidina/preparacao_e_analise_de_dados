import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# 1. Criar dados fictícios
# [glicose, imc]
X = np.array([
    [85, 22], [90, 24], [95, 25], [100, 26], [105, 27],
    [110, 28], [115, 29], [120, 24], [125, 26], [130, 28],
    [135, 29], [138, 30], [142, 31], [145, 32], [150, 33],
    [155, 34], [160, 35], [165, 36], [170, 37], [175, 38],
    [180, 39], [140, 29], [143, 28], [148, 31], [152, 30],
    [118, 31], [122, 32], [128, 33], [132, 34], [136, 35]
])

# 0 = sem risco, 1 = com risco
y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 0, 0, 1, 1,
    0, 0, 0, 0, 0
])


# 2. Dividir em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# 3. Criar e treinar o modelo
modelo = LogisticRegression()
modelo.fit(X_treino, y_treino)


# 4. Fazer previsões
previsoes = modelo.predict(X_teste)
probabilidades = modelo.predict_proba(X_teste)


# 5. Avaliar o modelo
print("Exactidão:", accuracy_score(y_teste, previsoes))
print(classification_report(y_teste, previsoes, target_names=["Sem Risco", "Com Risco"]))

# 6. Interpretar coeficientes
# Intercepto: valor quando todas as variáveis são 0
# Coeficientes: quanto cada variável influencia na previsão
print("Intercepto:", modelo.intercept_[0])
print("Coeficiente para glicose:", modelo.coef_[0][0])
print("Coeficiente para IMC:", modelo.coef_[0][1])

# Mostrar probabilidades previstas
print("\nProbabilidades previstas:")
for i in range(len(X_teste)):
    print(
        f"Paciente {i+1}: glicose={X_teste[i][0]}, imc={X_teste[i][1]}, "
        f"prob_sem_risco={probabilidades[i][0]:.3f}, "
        f"prob_com_risco={probabilidades[i][1]:.3f}, "
        f"classe_prevista={previsoes[i]}"
    )