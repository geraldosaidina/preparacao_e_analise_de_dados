import pandas as pd

df = pd.read_csv('vendas.csv', parse_dates=['data'])
df = df.set_index('data').sort_index()

# Extrair atributos temporais
df['dia_semana'] = df.index.dayofweek
df['mes'] = df.index.month
df['trimestre'] = df.index.quarter
df['e_fim_semana'] = (df.index.dayofweek >= 5).astype(int)

# Sazonalidade: dias ate Natal
df['dias_ate_natal'] = df.index.map(
    lambda d: (pd.Timestamp(d.year, 12, 25) - d).days % 365
)

# Media movel de 7 dias (lag feature)
df['media_7d'] = df['vendas'].shift(1).rolling(7).mean()

df.dropna(inplace=True)
print(df.head())