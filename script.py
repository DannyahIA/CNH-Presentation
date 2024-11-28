import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Título da aplicação
st.title("Análise de Dados - Grupo 1 (CSAT)")

# Upload do arquivo
uploaded_file = st.file_uploader("Faça upload do arquivo Excel:", type=["xlsx"])

if uploaded_file:
    # Carregar a folha "answers"
    data = pd.ExcelFile(uploaded_file)
    answers_df = data.parse("answers")
    
    # Filtrar colunas e Grupo 1
    columns_to_use = [
        "Grupo de Produto",
        "nota",
        "capacidade operacional (hectares por hora) (csat)",
        "adequação as diversas operações e implementos (csat)",
        "facilidade de operação (csat)",
        "conforto e ergonomia (csat)",
        "disponibilidade e confiabilidade mecânica  (csat)",
        "facilidade para realização de manutenções (csat)",
        "custo de manutenção (csat)",
        "consumo de combustível (litros por hectares) (csat)",
        "adaptabilidade as mais diversas condições de trabalho (csat)",
        "facilidade de uso do piloto automático (csat)",
        "geração e transmissão de dados para gestão da frota (csat)",
        "geração e transmissão de dados para gestão agrícola (csat)",
    ]
    filtered_df = answers_df[columns_to_use]
    grupo1_df = filtered_df[filtered_df["Grupo de Produto"] == "Grupo 1"]

    st.write("### Dados Filtrados (Grupo 1)")
    st.dataframe(grupo1_df)

    # Gráfico de Moda
    st.write("### Gráfico de Moda")
    csat_columns = grupo1_df.columns[2:]
    moda_values = grupo1_df[csat_columns].mode().iloc[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=moda_values.index, y=moda_values.values, palette="viridis", ax=ax)
    ax.set_title("Moda dos Valores CSAT - Grupo 1")
    ax.set_ylabel("Moda")
    ax.set_xlabel("Indicadores CSAT")
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Mapa de Covariância
    st.write("### Mapa de Covariância")
    cov_matrix = grupo1_df[csat_columns].cov()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    ax.set_title("Matriz de Covariância - Indicadores CSAT (Grupo 1)")
    st.pyplot(fig)

    # Métodos Não Supervisionados - K-Means e Expectation Maximization
    st.write("### Métodos Não Supervisionados")
    k = st.slider("Selecione o número de clusters:", min_value=2, max_value=10, value=3)

    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    grupo1_df["KMeans Cluster"] = kmeans.fit_predict(grupo1_df[csat_columns].fillna(0))
    st.write("#### Clusters (K-Means)")
    st.dataframe(grupo1_df[["nota", "KMeans Cluster"]])

    # Expectation Maximization (Gaussian Mixture)
    gmm = GaussianMixture(n_components=k, random_state=42)
    grupo1_df["EM Cluster"] = gmm.fit_predict(grupo1_df[csat_columns].fillna(0))
    st.write("#### Clusters (Expectation Maximization)")
    st.dataframe(grupo1_df[["nota", "EM Cluster"]])

    # Métodos Supervisionados - Random Forests e KNN
    st.write("### Métodos Supervisionados")
    target = st.selectbox("Selecione a variável alvo:", ["nota"])
    X = grupo1_df[csat_columns].fillna(0)
    y = grupo1_df[target]

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Random Forests
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    st.write("#### Random Forests - Relatório de Classificação")
    st.text(classification_report(y_test, rf_predictions))

    # KNN
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    st.write("#### KNN - Relatório de Classificação")
    st.text(classification_report(y_test, knn_predictions))
