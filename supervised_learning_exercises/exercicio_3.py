import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
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

nomes_variaveis = ["glicose", "imc"]
nomes_classes = ["Sem Risco", "Com Risco"]

# 3. Árvore com max_depth=3
arvore_3 = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

arvore_3.fit(X_treino, y_treino)

# Previsões
prev_treino_3 = arvore_3.predict(X_treino)
prev_teste_3 = arvore_3.predict(X_teste)

# Exactidão
acc_treino_3 = accuracy_score(y_treino, prev_treino_3)
acc_teste_3 = accuracy_score(y_teste, prev_teste_3)

print("=== Árvore com max_depth=3 ===")
print("Exactidão no treino:", acc_treino_3)
print("Exactidão no teste:", acc_teste_3)
print(classification_report(y_teste, prev_teste_3, target_names=nomes_classes))

# Visualizar árvore
plt.figure(figsize=(12, 7))
plot_tree(
    arvore_3,
    feature_names=nomes_variaveis,
    class_names=nomes_classes,
    filled=True,
    rounded=True
)
plt.title("Árvore de Decisão com max_depth=3")
plt.show()

# 4. Árvore com max_depth=10
arvore_10 = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    random_state=42
)

arvore_10.fit(X_treino, y_treino)

# Previsões
prev_treino_10 = arvore_10.predict(X_treino)
prev_teste_10 = arvore_10.predict(X_teste)

# Exactidão
acc_treino_10 = accuracy_score(y_treino, prev_treino_10)
acc_teste_10 = accuracy_score(y_teste, prev_teste_10)

print("=== Árvore com max_depth=10 ===")
print("Exactidão no treino:", acc_treino_10)
print("Exactidão no teste:", acc_teste_10)
print(classification_report(y_teste, prev_teste_10, target_names=nomes_classes))

# Visualizar árvore
plt.figure(figsize=(16, 9))
plot_tree(
    arvore_10,
    feature_names=nomes_variaveis,
    class_names=nomes_classes,
    filled=True,
    rounded=True
)
plt.title("Árvore de Decisão com max_depth=10")
plt.show()