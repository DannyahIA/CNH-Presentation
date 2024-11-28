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

st.markdown("""
### Perguntas e Respostas

**1. Quais perguntas do CSI mais afetam/influenciam o cliente a ser um Neutro?**

*Com base na análise dos dados, as perguntas relacionadas a fatores como facilidade de operação, conforto e ergonomia, disponibilidade e confiabilidade mecânica e facilidade para a realização de manutenções foram as que mais influenciaram os clientes a se tornarem Neutros. Os neutros tendem a pontuar essas questões de forma moderada, indicando que, embora não estejam completamente satisfeitos, também não se sentem insatisfeitos o suficiente para se classificar como detratores.*

**2. Quais perguntas do CSI mais afetam/influenciam o cliente a ser um Detrator?**

*Para os Detratores, as perguntas que mais afetam a pontuação incluem questões como custo de manutenção, consumo de combustível, adequação a diversas operações e implementos, e facilidade de uso do piloto automático. Esses fatores parecem ser determinantes para a insatisfação, com muitos detratores apontando essas áreas como grandes fontes de frustração. Clientes insatisfeitos frequentemente relatam problemas nessas áreas como motivos para dar notas baixas.*

**3. Com base em qual argumento/critérios vocês definiram as perguntas que mais afetam o NPS?**

*A definição das perguntas mais impactantes para o NPS foi baseada em uma análise quantitativa das médias de nota e da distribuição das pontuações nas diferentes áreas. Além disso, observamos as correlações entre as variáveis de satisfação, como facilidade de operação e manutenção. Aquelas que mostraram uma maior variação nas respostas, ou que apresentaram um impacto direto nas categorias de Neutros e Detratores, foram selecionadas como as mais influentes para o NPS.*

**4. Com base em tudo que foi analisado, qual sugestão de melhoria no produto vocês dariam para melhorar o NPS?**

*A partir da análise dos dados, uma sugestão fundamental para melhorar o NPS seria focar em reduzir o custo de manutenção e melhorar o consumo de combustível. Além disso, aprimorar a facilidade de operação e a ergonomia do produto pode aumentar a satisfação, especialmente entre os neutros, que apresentaram uma pontuação moderada nessas áreas. Trabalhar para tornar o produto mais adaptável e reduzir as queixas sobre o piloto automático também pode ajudar a converter detratores em promotores. Um investimento em melhoria contínua nessas áreas poderia gerar um impacto significativo no NPS geral.*
""")

# Upload do arquivo
uploaded_file = st.file_uploader("Faça upload do arquivo Excel(Lista NPS):", type=["xlsx"])

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
        target = st.selectbox("Selecione a variável alvo (para classificação):", ["nota"])
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

    # Criar abas para neutros e detratores
    tab1, tab2 = st.tabs(["Neutros", "Detratores"])

    with tab1:
        plot_data(neutros_df, "Neutros")

    with tab2:
        plot_data(detratores_df, "Detratores")
