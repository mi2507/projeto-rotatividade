# 📦 Streamlit App (Versão Final Corrigida) - Previsão de Rotatividade com Gráficos
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Configuração inicial
st.set_page_config(page_title="Previsão de Rotatividade", layout="centered")
st.title("🔍 Previsão de Rotatividade de Funcionários")
st.write("Preencha os dados abaixo para prever se o funcionário pode sair da empresa. "
         "O resultado virá com probabilidade e gráficos comparativos.")

# Carregar modelo e base
# modelo = joblib.load('modelo_xgboost_final.joblib')
with open('modelo_xgboost_final.joblib', 'rb') as f:
    modelo = pickle.load(f)
df_empresa = pd.read_csv('rh_data.csv')

# Mapas amigáveis
map_education = {"Ensino Médio ou abaixo": 1, "Graduação": 2, "Bacharelado": 3, "Mestrado": 4, "Doutorado": 5}
map_joblevel = {"Júnior": 1, "Pleno": 2, "Sênior": 3, "Gerente": 4, "Diretor": 5}
map_stock = {"Nenhuma": 0, "Baixa": 1, "Média": 2, "Alta": 3}
map_gender = {"Masculino": "Male", "Feminino": "Female"}
map_marital = {"Solteiro(a)": "Single", "Casado(a)": "Married", "Divorciado(a)": "Divorced"}
map_travel = {"Não viaja": "Non-Travel", "Viaja raramente": "Travel_Rarely", "Viaja frequentemente": "Travel_Frequently"}
map_department = {"Recursos Humanos": "Human Resources", "Pesquisa e Desenvolvimento": "Research & Development", "Vendas": "Sales"}
map_jobrole = {
    "Executivo de Vendas": "Sales Executive",
    "Cientista Pesquisador": "Research Scientist",
    "Técnico de Laboratório": "Laboratory Technician",
    "Diretor de Manufatura": "Manufacturing Director",
    "Representante de Saúde": "Healthcare Representative",
    "Gerente": "Manager",
    "Representante de Vendas": "Sales Representative",
    "Diretor de Pesquisa": "Research Director",
    "Recursos Humanos": "Human Resources"
}

# Formulário
with st.form("form_funcionario"):
    st.subheader("Informações Pessoais")
    col1, col2, col3 = st.columns(3)
    with col1: Age = st.slider("Idade", 18, 65, 30)
    with col2: Gender = st.selectbox("Gênero", list(map_gender.keys()))
    with col3: MaritalStatus = st.selectbox("Estado Civil", list(map_marital.keys()))

    st.subheader("Dados Profissionais")
    col4, col5, col6 = st.columns(3)
    with col4: Education = st.selectbox("Nível de Educação", list(map_education.keys()))
    with col5: JobLevel = st.selectbox("Nível de Cargo", list(map_joblevel.keys()))
    with col6: StockOptionLevel = st.selectbox("Opções de Ações", list(map_stock.keys()))

    Department = st.selectbox("Departamento", list(map_department.keys()))
    JobRole = st.selectbox("Cargo", list(map_jobrole.keys()))
    BusinessTravel = st.selectbox("Viagem a Trabalho", list(map_travel.keys()))

    st.subheader("Dados Financeiros e Tempo na Empresa")
    col7, col8, col9 = st.columns(3)
    with col7:
        MonthlyIncome = st.number_input("Salário Mensal (R$)", min_value=500, max_value=20000, value=4000)
        PercentSalaryHike = st.slider("% Aumento Salarial", 0, 50, 15)
    with col8:
        NumCompaniesWorked = st.slider("Nº de Empresas Anteriores", 0, 10, 2)
        TotalWorkingYears = st.slider("Anos de Experiência Total", 0, 40, 10)
        DistanceFromHome = st.slider("Distância de Casa (km)", 1, 100, 10)
    with col9:
        YearsAtCompany = st.slider("Anos na Empresa", 0, 40, 4)
        YearsSinceLastPromotion = st.slider("Anos desde última promoção", 0, 15, 2)
        YearsWithCurrManager = st.slider("Anos com atual gerente", 0, 15, 3)
        TrainingTimesLastYear = st.slider("Treinamentos no último ano", 0, 6, 2)

    submitted = st.form_submit_button("🔮 Prever Resultado")

if submitted:
    # Mapeamento para os valores usados no modelo
    Education_val = map_education[Education]
    JobLevel_val = map_joblevel[JobLevel]
    Stock_val = map_stock[StockOptionLevel]
    Gender_val = map_gender[Gender]
    Marital_val = map_marital[MaritalStatus]
    Travel_val = map_travel[BusinessTravel]
    Dept_val = map_department[Department]
    JobRole_val = map_jobrole[JobRole]

    # Criar DataFrame para previsão
    dados = pd.DataFrame({
        'Age': [Age],
        'DistanceFromHome': [DistanceFromHome],
        'Education': [Education_val],
        'Gender': [Gender_val],
        'JobLevel': [JobLevel_val],
        'MonthlyIncome': [MonthlyIncome],
        'NumCompaniesWorked': [NumCompaniesWorked],
        'PercentSalaryHike': [PercentSalaryHike],
        'StockOptionLevel': [Stock_val],
        'TotalWorkingYears': [TotalWorkingYears],
        'TrainingTimesLastYear': [TrainingTimesLastYear],
        'YearsAtCompany': [YearsAtCompany],
        'YearsSinceLastPromotion': [YearsSinceLastPromotion],
        'YearsWithCurrManager': [YearsWithCurrManager],
        'BusinessTravel_' + Travel_val: [1],
        'Department_' + Dept_val: [1],
        'EducationField_Life Sciences': [0],
        'JobRole_' + JobRole_val.replace(" ", "_"): [1],
        'MaritalStatus_' + Marital_val: [1],
    })

    # Completar colunas ausentes
    for col in modelo.get_booster().feature_names:
        if col not in dados.columns:
            dados[col] = 0
    dados = dados[modelo.get_booster().feature_names]

    # Fazer previsão
    prob = modelo.predict_proba(dados)[0][1]
    pred = modelo.predict(dados)[0]

    resultado = "⚠️ Funcionário pode SAIR da empresa." if pred == 1 else "✅ Funcionário deve PERMANECER na empresa."

    st.markdown("---")
    st.subheader("Resultado da Previsão")
    st.write(f"**Probabilidade de saída:** {prob*100:.2f}%")
    st.success(resultado)

    # --- GRÁFICO DE PROBABILIDADE ---
    st.subheader("Probabilidade de Saída (Visual)")
    fig_prob, ax_prob = plt.subplots(figsize=(6, 1.2))
    ax_prob.barh([0], [prob*100], color='#6c337c')  # Roxo
    ax_prob.set_xlim(0, 100)
    ax_prob.set_yticks([])
    ax_prob.set_xlabel('Probabilidade de Saída (%)')
    ax_prob.bar_label(ax_prob.containers[0], fmt='%.2f%%')
    st.pyplot(fig_prob, use_container_width=True)

    # --- GRÁFICO COMPARATIVO ---
    st.markdown("### Comparação com a Média de Funcionários da Empresa")

    atributos = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'DistanceFromHome']
    labels = ['Idade (anos)', 'Salário Mensal (R$)', 'Experiência (anos)', 'Anos na Empresa', 'Distância (km)']
    valores_func = [Age, MonthlyIncome, TotalWorkingYears, YearsAtCompany, DistanceFromHome]
    medias = [df_empresa[a].mean() for a in atributos]

    # Calcular percentuais relativos à média
    percentuais_func = [(valores_func[i] / medias[i]) * 100 if medias[i] > 0 else 0 for i in range(len(atributos))]
    percentuais_media = [100 for _ in medias]  # 100% sempre representa a média

    # Plotar gráfico horizontal com rótulos
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(atributos))
    ax.barh(y - 0.2, percentuais_media, height=0.4, label='Média Funcionários', color='green')
    bars = ax.barh(y + 0.2, percentuais_func, height=0.4, label='Funcionário', color='#6a0dad')

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Percentual em Relação à Média (%)')
    ax.set_xlim(0, max(120, max(percentuais_func) * 1.1))
    ax.legend()

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}%", va='center')

    st.pyplot(fig, use_container_width=True)

  
