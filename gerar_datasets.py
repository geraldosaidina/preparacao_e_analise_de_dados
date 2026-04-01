import numpy as np
import pandas as pd

np.random.seed(42)

# =========================================================
# CENARIO 1. Limpeza de Dados de Saude
# Colunas: idade, peso, pressao, glicemia, diagnostico
# =========================================================
n1 = 1200

idade = np.random.randint(18, 81, n1).astype(float)
peso = np.round(np.random.normal(70, 12, n1), 1)
pressao = np.round(np.random.normal(125, 18, n1), 1)
glicemia = np.round(np.random.normal(105, 25, n1), 1)

# diagnostico simples baseado em risco
diagnostico = (
    (idade > 50).astype(int)
    + (pressao > 140).astype(int)
    + (glicemia > 126).astype(int)
)
diagnostico = np.where(diagnostico >= 2, "alto_risco", "baixo_risco")

df1 = pd.DataFrame({
    "idade": idade,
    "peso": peso,
    "pressao": pressao,
    "glicemia": glicemia,
    "diagnostico": diagnostico
})

# 15% glicemia em falta
idx_glicemia = np.random.choice(df1.index, size=int(0.15 * n1), replace=False)
df1.loc[idx_glicemia, "glicemia"] = np.nan

# 8% pressao em falta
idx_pressao = np.random.choice(df1.index, size=int(0.08 * n1), replace=False)
df1.loc[idx_pressao, "pressao"] = np.nan

# 3 idades negativas
idx_idade_neg = np.random.choice(df1.index, size=3, replace=False)
df1.loc[idx_idade_neg, "idade"] = [-5, -12, -1]

df1.to_csv("pacientes.csv", index=False)

# =========================================================
# CENARIO 2. Deteccao de Outliers em Dados Financeiros
# Colunas: valor, hora, tipo, saldo
# =========================================================
n2 = 50000

valor = np.round(np.random.lognormal(mean=7.5, sigma=0.8, size=n2), 2)
hora = np.random.randint(0, 24, n2)
tipo = np.random.choice(
    ["transferencia", "levantamento", "pagamento", "deposito"],
    size=n2,
    p=[0.35, 0.20, 0.30, 0.15]
)
saldo = np.round(np.random.normal(80000, 35000, n2), 2)

df2 = pd.DataFrame({
    "valor": valor,
    "hora": hora,
    "tipo": tipo,
    "saldo": saldo
})

# alguns valores negativos impossiveis
idx_neg = np.random.choice(df2.index, size=25, replace=False)
df2.loc[idx_neg, "valor"] = -np.abs(df2.loc[idx_neg, "valor"])

# alguns outliers 50x acima do normal
idx_out = np.random.choice(df2.index.difference(idx_neg), size=40, replace=False)
df2.loc[idx_out, "valor"] = df2["valor"].median() * 50

df2.to_csv("transaccoes.csv", index=False)

# =========================================================
# CENARIO 3. Codificacao de Variaveis Categoricas
# Colunas: regiao, contrato, plano, churn
# =========================================================
n3 = 2000

regiao = np.random.choice(["Maputo", "Beira", "Nampula", "Tete"], size=n3)
contrato = np.random.choice(["mensal", "semestral", "anual"], size=n3, p=[0.55, 0.25, 0.20])
plano = np.random.choice(["Basico", "Standard", "Premium"], size=n3, p=[0.4, 0.35, 0.25])

# churn mais provavel em contrato mensal e plano basico
churn_prob = []
for c, p in zip(contrato, plano):
    prob = 0.15
    if c == "mensal":
        prob += 0.20
    if p == "Basico":
        prob += 0.15
    if p == "Premium":
        prob -= 0.05
    churn_prob.append(prob)

churn = np.where(np.random.rand(n3) < np.array(churn_prob), "Sim", "Nao")

df3 = pd.DataFrame({
    "regiao": regiao,
    "contrato": contrato,
    "plano": plano,
    "churn": churn
})

df3.to_csv("clientes_telecom.csv", index=False)

# =========================================================
# CENARIO 4. Normalizacao de Dados para KNN
# Colunas: idade, salario, divida, risco
# =========================================================
n4 = 3000

idade = np.random.randint(18, 66, n4)
salario = np.round(np.random.uniform(5000, 200000, n4), 2)
divida = np.round(np.random.uniform(0, 500000, n4), 2)

# regra simples para risco
score = (divida / (salario + 1)) + (idade / 100)
risco = np.where(score > np.median(score), 1, 0)

df4 = pd.DataFrame({
    "idade": idade,
    "salario": salario,
    "divida": divida,
    "risco": risco
})

df4.to_csv("credito_knn.csv", index=False)

# =========================================================
# CENARIO 5. Feature Engineering em Dados Temporais
# Colunas: data, vendas
# =========================================================
datas = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
n5 = len(datas)

base = 20000
sazonal_mensal = 3000 * np.sin(2 * np.pi * np.arange(n5) / 30)
efeito_fim_semana = np.where(datas.dayofweek >= 5, 2500, 0)
efeito_natal = np.where((datas.month == 12) & (datas.day >= 15), 5000, 0)
ruido = np.random.normal(0, 1200, n5)

vendas = np.round(base + sazonal_mensal + efeito_fim_semana + efeito_natal + ruido, 2)

df5 = pd.DataFrame({
    "data": datas,
    "vendas": vendas
})

df5.to_csv("vendas.csv", index=False)

# =========================================================
# CENARIO 6. Pipeline Completo End-to-End
# Colunas numericas e categoricas com missing values
# =========================================================
n6 = 4000

idade = np.random.randint(21, 70, n6).astype(float)
rendimento = np.round(np.random.normal(75000, 25000, n6), 2)
anos_emprego = np.round(np.random.uniform(0, 30, n6), 1)
montante_emprestimo = np.round(np.random.uniform(5000, 300000, n6), 2)
taxa_juro = np.round(np.random.uniform(5, 32, n6), 2)
percentagem_rendimento = np.round(montante_emprestimo / np.maximum(rendimento, 1), 3)
historico_credito = np.round(np.random.uniform(1, 20, n6), 1)

habitacao = np.random.choice(["arrendada", "propria", "hipoteca", "outra"], n6)
finalidade_emprestimo = np.random.choice(
    ["educacao", "saude", "negocio", "pessoal", "habitacao"],
    n6
)
classe_emprestimo = np.random.choice(["A", "B", "C", "D"], n6, p=[0.30, 0.30, 0.25, 0.15])
historico_incumprimento = np.random.choice(["sim", "nao"], n6, p=[0.15, 0.85])

# alvo binario
risco_total = (
    (percentagem_rendimento > 2.0).astype(int)
    + (taxa_juro > 20).astype(int)
    + (historico_incumprimento == "sim").astype(int)
    + np.isin(classe_emprestimo, ["C", "D"]).astype(int)
)
estado_emprestimo = np.where(risco_total >= 2, 1, 0)

df6 = pd.DataFrame({
    "idade": idade,
    "rendimento": rendimento,
    "anos_emprego": anos_emprego,
    "montante_emprestimo": montante_emprestimo,
    "taxa_juro": taxa_juro,
    "percentagem_rendimento": percentagem_rendimento,
    "historico_credito": historico_credito,
    "habitacao": habitacao,
    "finalidade_emprestimo": finalidade_emprestimo,
    "classe_emprestimo": classe_emprestimo,
    "historico_incumprimento": historico_incumprimento,
    "estado_emprestimo": estado_emprestimo
})

# missing values artificiais
for col, frac in [("taxa_juro", 0.08), ("anos_emprego", 0.05), ("habitacao", 0.04)]:
    idx = np.random.choice(df6.index, size=int(frac * n6), replace=False)
    df6.loc[idx, col] = np.nan

# alguns outliers
idx_outlier = np.random.choice(df6.index, size=20, replace=False)
df6.loc[idx_outlier, "montante_emprestimo"] *= 5

df6.to_csv("credito_pipeline.csv", index=False)

print("Datasets ficticios gerados com sucesso.")