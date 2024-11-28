from sklearn.tree import plot_tree
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.decomposition import PCA
import numpy as np

# Título da aplicação
st.title("Dashboard - Análise de Dados")

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

    # Filtrar neutros e detratores
    neutros_df = grupo1_df[(grupo1_df["nota"] >= 7) & (grupo1_df["nota"] <= 8)]
    detratores_df = grupo1_df[grupo1_df["nota"] <= 6]

    def plot_data(df, title_suffix):
        st.write(f"### Dados Filtrados (Grupo 1 - {title_suffix})")
        st.dataframe(df)

        # Cálculo da Moda
        st.write(f"### Estatísticas: Moda ({title_suffix})")
        moda = df.mode().iloc[0]  # A moda de cada coluna
        st.write(moda)

        # Gráfico de Moda
        st.write(f"#### Gráfico de Moda ({title_suffix})")
        moda_numeric = moda[2:]  # Ignorar as duas primeiras colunas não numéricas
        fig, ax = plt.subplots()
        moda_numeric.plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title(f"Moda das Variáveis CSAT ({title_suffix})")
        ax.set_ylabel("Valores")
        ax.set_xticklabels(moda_numeric.index, rotation=45, ha="right")
        st.pyplot(fig)

        # Cálculo da Covariância
        st.write(f"### Estatísticas: Covariância ({title_suffix})")
        csat_columns = df.columns[2:]  # Apenas colunas numéricas
        cov_matrix = df[csat_columns].cov()
        st.write(cov_matrix)

        # Visualização da Matriz de Covariância
        st.write(f"#### Mapa de Calor da Covariância ({title_suffix})")
        fig, ax = plt.subplots()
        sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Métodos Não Supervisionados
        st.write(f"### Métodos Não Supervisionados ({title_suffix})")
        k = st.slider("Selecione o número de clusters:", min_value=2, max_value=10, value=3, key=f"slider_{title_suffix}")
        X = df[csat_columns].fillna(0)

        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["KMeans Cluster"] = kmeans.fit_predict(X)

        # Redução de Dimensionalidade para Visualização
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(X)
        df["PCA1"] = reduced_data[:, 0]
        df["PCA2"] = reduced_data[:, 1]

        fig, ax = plt.subplots()
        sns.scatterplot(
            x="PCA1", y="PCA2", hue="KMeans Cluster", palette="viridis", data=df, ax=ax
        )
        ax.set_title(f"Clusters Formados - K-Means (PCA) ({title_suffix})")
        st.pyplot(fig)

        # Gaussian Mixture (Expectation Maximization)
        gmm = GaussianMixture(n_components=k, random_state=42)
        df["EM Cluster"] = gmm.fit_predict(X)

        fig, ax = plt.subplots()
        sns.scatterplot(
            x="PCA1", y="PCA2", hue="EM Cluster", palette="coolwarm", data=df, ax=ax
        )
        ax.set_title(f"Clusters Formados - Expectation Maximization (PCA) ({title_suffix})")
        st.pyplot(fig)

        # Métodos Supervisionados
        st.write(f"### Métodos Supervisionados ({title_suffix})")
        target = st.selectbox("Selecione a variável alvo (para classificação):", ["nota"], key=f"selectbox_{title_suffix}")
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Random Forests
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)

        st.write(f"#### Random Forests - Árvore de Decisão ({title_suffix})")
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(rf_model.estimators_[0], feature_names=csat_columns, class_names=rf_model.classes_.astype(str), filled=True, ax=ax)
        st.pyplot(fig)

        # Matriz de Confusão
        st.write(f"#### Random Forests - Matriz de Confusão ({title_suffix})")
        rf_cm = confusion_matrix(y_test, rf_predictions)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(rf_cm, display_labels=rf_model.classes_).plot(ax=ax, cmap="Blues")
        st.pyplot(fig)

        # KNN
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        knn_predictions = knn_model.predict(X_test)

        st.write(f"#### KNN - Matriz de Confusão ({title_suffix})")
        knn_cm = confusion_matrix(y_test, knn_predictions)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(knn_cm, display_labels=knn_model.classes_).plot(ax=ax, cmap="Oranges")
        st.pyplot(fig)

        # Relatório de Classificação
        st.write(f"#### Relatórios de Classificação ({title_suffix})")
        st.write("**Random Forests**")
        st.text(classification_report(y_test, rf_predictions))

        st.write("**KNN**")
        st.text(classification_report(y_test, knn_predictions))
        # Gráfico de Clusters para Gaussian Mixture
        st.write(f"#### Gráfico de Clusters - Gaussian Mixture ({title_suffix})")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x="PCA1", y="PCA2", hue="EM Cluster", palette="coolwarm", data=df, ax=ax
        )
        ax.set_title(f"Clusters Formados - Gaussian Mixture (PCA) ({title_suffix})")
        st.pyplot(fig)

    # Plot data for neutros
    plot_data(neutros_df, "Neutros")

    # Plot data for detratores
    plot_data(detratores_df, "Detratores")
