# Importacoes necessarias
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

#Carregar o conjunto de dados Iris (classico em ML)
dados = load_iris()
x, y = dados.data, dados.target

# PASSO OBRIGATORIO: Normalizar as variaveis
normalizador = StandardScaler()
X_normalizado = normalizador.fit_transform(x)

# Dividir em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X_normalizado, y, test_size=0.2, random_state=42)

# Testar diferentes valores de K
for k in [1, 3, 5, 7, 11]:
    modelo_knn = KNeighborsClassifier(n_neighbors=k)
    pontuacoes = cross_val_score (modelo_knn, X_treino, y_treino, cv=5)
    print(f'K={k}: Exactidao media = {pontuacoes.mean() :.4f}')
# Treinar com o melhor K encontrado
melhor_k = 5
modelo_knn = KNeighborsClassifier(n_neighbors=melhor_k, metric='euclidean')
modelo_knn.fit(X_treino, y_treino)
print('Exactidao no teste: ', accuracy_score(y_teste, modelo_knn.predict(X_teste)))
