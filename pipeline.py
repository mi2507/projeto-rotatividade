# pipeline.py - Re-treino automático do modelo de Rotatividade

import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# --- 1. Carregar os dados ---
print("🔄 Carregando base de dados...")
df = pd.read_csv('rh_data.csv')

# --- 2. Limpeza e pré-processamento (mesmo que no notebook) ---
print("🧹 Limpando e transformando dados...")
df['NumCompaniesWorked'].fillna(df['NumCompaniesWorked'].median(), inplace=True)
df['TotalWorkingYears'].fillna(df['TotalWorkingYears'].median(), inplace=True)

colunas_para_remover = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeID']
df.drop(columns=colunas_para_remover, inplace=True, errors='ignore')

# Variável alvo (0 = Não Saiu, 1 = Saiu)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Criar variáveis derivadas
df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 29, 50, 65], labels=['Jovem', 'Adulto', 'Senior'])
df['YearsAtCompanyCat'] = pd.cut(df['YearsAtCompany'], bins=[-1, 3, 7, 40], labels=['Curto', 'Médio', 'Longo'])
df['DistanceFromHomeCat'] = pd.cut(df['DistanceFromHome'], bins=[-1, 5, 15, 100], labels=['Curta', 'Média', 'Longa'])
df['LongTimeNoPromotion'] = (df['YearsSinceLastPromotion'] > 5).astype(int)
df['NumCompaniesWorkedCat'] = pd.cut(df['NumCompaniesWorked'], bins=[-1, 2, 5, 20], labels=['Poucas', 'Médias', 'Muitas'])

# Dividir X e y
y = df['Attrition']
X = df.drop('Attrition', axis=1)

# Codificar variáveis categóricas
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# --- 3. Dividir em treino e teste ---
print("✂️ Dividindo em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Treinar modelo ---
print("🚀 Treinando modelo XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# --- 5. Avaliar modelo ---
print("📊 Avaliando desempenho...")
y_pred = xgb_model.predict(X_test)
metrics = {
    "acuracia": round(accuracy_score(y_test, y_pred), 4),
    "precisao": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1": round(f1_score(y_test, y_pred), 4)
}
print("Métricas:", metrics)

# --- 6. Salvar modelo e métricas ---
print("💾 Salvando modelo e métricas...")
joblib.dump(xgb_model, 'modelo_xgboost_final.joblib')

with open('metricas_modelo.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("✅ Pipeline concluído! Modelo e métricas atualizados.")
