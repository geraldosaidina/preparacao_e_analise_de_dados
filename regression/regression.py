# Importacoes necessárias
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
# Dados de exemplo: [frequencia da palavra 'oferta', numero de links]
X = np.array([[0.1, 1], [0.9, 10], [0.2, 2], [0.8, 8], [0.05, 0], [0.7, 7]])
Y =np.array([0, 1, 0, 1, 0, 1])  # 0 = não-spam, 1 = spam

# Dividir em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
X, Y, test_size=0.3, random_state=42)

#Criar e treinar o modelo
modelo = LogisticRegression()
modelo.fit (X_treino, y_treino)

#Fazer previsões
previsoes = modelo.predict(X_teste)
probabilidades = modelo.predict_proba(X_teste)

# Avaliar o modelo
print ('Exactidão: ', accuracy_score (y_teste, previsoes))

print (classification_report (y_teste, previsoes, target_names=['Nao-Spam', 'Spam' ]))