import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path(__file__).resolve().parent / "transaccoes.csv"
df = pd.read_csv(csv_path)

# 1. Remover valores impossiveis (negativos)
df = df[df['valor'] >= 0]

# 2. Deteccao por IQR
Q1 = df['valor'].quantile(0.25)
Q3 = df['valor'].quantile(0.75)
IQR = Q3 - Q1
limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR

# 3. Winsorizing: substituir outliers pelos limites
df['valor_clean'] = df['valor'].clip(lower=limite_inf, upper=limite_sup)

print(f'Outliers: {(df["valor"] > limite_sup).sum()}')
print(df.head())