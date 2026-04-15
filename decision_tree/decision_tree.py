# Importacoes necessarias
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#Carregar dados
dados = load_iris()

x, y = dados.data, dados.target
nomes_classes = dados.target_names
nomes_variaveis = dados.feature_names
X_treino, X_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# Modelo com controlo de sobreajusta
arvore = DecisionTreeClassifier(
    criterion='gini', # ou 'entropy' para Ganho de Informacao
    max_depth=4, #profundidade maxima da arvore
    min_samples_split=5, 
    min_samples_leaf=2,
    random_state=42
)

# minimo de exemplos em cada foll
arvore.fit(X_treino, y_treino)

# Avaliar 
previsoes = arvore.predict(X_teste)
print('Exactidão: ', accuracy_score(y_teste, previsoes))
print(classification_report(y_teste, previsoes, target_names=nomes_classes))

# Visualizar a estrutura da arvore em texto
print (export_text (arvore, feature_names=list(nomes_variaveis)))

# Visualizar a arvore graficamente
plt.figure(figsize=(16, 8))
plot_tree(arvore, feature_names=nomes_variaveis, class_names=nomes_classes, filled=True, rounded=True)
plt.savefig('arvore_decisao.png', bbox_inches='tight')
plt.show()

# Importancia das variaveis
for var, imp in zip(nomes_variaveis, arvore.feature_importances_) :
    print(f'{var}: {imp :.4f} ')