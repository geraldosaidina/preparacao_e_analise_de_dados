import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('credito_pipeline.csv')

X = df.drop(columns=['estado_emprestimo'])
y = df['estado_emprestimo']

num_cols = [
    'idade',
    'rendimento',
    'anos_emprego',
    'montante_emprestimo',
    'taxa_juro',
    'percentagem_rendimento',
    'historico_credito'
]

cat_cols = [
    'habitacao',
    'finalidade_emprestimo',
    'classe_emprestimo',
    'historico_incumprimento'
]

num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc', StandardScaler())
])

cat_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('enc', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100))
])

scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
print(f'F1: {scores.mean():.3f} +/- {scores.std():.3f}')