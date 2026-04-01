import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Carregar dados
df = pd.read_csv('pacientes.csv')

# 1. Detectar e corrigir erros de idade
df['idade'] = df['idade'].apply(lambda x: np.nan if x < 0 else x)
df['idade'].fillna(df['idade'].median(), inplace=True)

# 2. Imputar pressao arterial (MCAR) - mediana e robusta
imp_median = SimpleImputer(strategy='median')
df['pressao'] = imp_median.fit_transform(df[['pressao']])

# 3. Imputar glicemia (MAR) - KNN usa variaveis correlacionadas
knn_imp = KNNImputer(n_neighbors=5)
df[['glicemia', 'idade', 'peso']] = knn_imp.fit_transform(df[['glicemia', 'idade', 'peso']])

print(df.isnull().sum())  # Verificar: 0 missing values
print(df.head())