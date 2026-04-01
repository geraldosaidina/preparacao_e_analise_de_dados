import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('clientes_telecom.csv')

# 1. One-Hot para 'regiao' (nominal)
# drop_first=True evita multicolinearidade
df = pd.get_dummies(df, columns=['regiao'], drop_first=True)

# 2. Ordinal para 'contrato' (ordem natural)
ordem_contrato = [['mensal', 'semestral', 'anual']]
enc = OrdinalEncoder(categories=ordem_contrato)
df[['contrato']] = enc.fit_transform(df[['contrato']])

# 3. Ordinal para 'plano'
ordem_plano = [['Basico', 'Standard', 'Premium']]
enc2 = OrdinalEncoder(categories=ordem_plano)
df[['plano']] = enc2.fit_transform(df[['plano']])

print(df.head())