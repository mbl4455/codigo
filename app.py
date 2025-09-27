# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("🎬 Análise de Filmes - TMDb 5000")

# --- Leitura dos dados ---
@st.cache_data
def carregar_dados():
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    credits_df = pd.read_csv("tmdb_5000_credits.csv", engine='python', on_bad_lines='skip')
    df = movies_df.merge(credits_df, left_on="id", right_on="movie_id")
    df.drop("movie_id", axis=1, inplace=True)
    return df

df = carregar_dados()

# --- Função para tratar colunas JSON ---
def parse_json_column(df, column):
    for index, row in df.iterrows():
        try:
            df.at[index, column] = json.loads(row[column])
        except (TypeError, json.JSONDecodeError):
            df.at[index, column] = []
    return df

for col in ['genres', 'keywords', 'cast', 'crew']:
    df = parse_json_column(df, col)

# --- Adiciona o gênero principal ---
def main_genre(genres):
    if genres:
        return genres[0]["name"]
    return "Unknown"

df["main_genre"] = df["genres"].apply(main_genre)

# --- Layout visual com abas ---
aba = st.sidebar.radio("Navegação", ["Exploração", "Gráficos", "Modelo Preditivo"])

# ---------------------------
# 📊 Aba: Exploração de Dados
# ---------------------------
if aba == "Exploração":
    st.subheader("📌 Estatísticas Descritivas")
    st.dataframe(df[["budget", "revenue", "vote_average", "popularity"]].describe())

    st.subheader("🎯 Top 5 filmes por nota")
    top_movies = df.sort_values("vote_average", ascending=False)[["title_x", "vote_average", "vote_count"]].head(5)
    st.dataframe(top_movies)

# ---------------------------
# 📈 Aba: Visualizações
# ---------------------------
elif aba == "Gráficos":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição das Notas")
        fig, ax = plt.subplots()
        sns.histplot(df["vote_average"], kde=True, ax=ax, color='skyblue')
        st.pyplot(fig)

    with col2:
        st.subheader("Dispersão: Orçamento vs Receita")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="budget", y="revenue", ax=ax)
        st.pyplot(fig)

    st.subheader("Boxplot: Nota Média por Gênero")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="main_genre", y="vote_average", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Quantidade de Filmes por Idioma")
    fig, ax = plt.subplots()
    sns.countplot(data=df, y="original_language", order=df["original_language"].value_counts().index, ax=ax)
    st.pyplot(fig)

# ---------------------------
# 🤖 Aba: Modelo de Previsão
# ---------------------------
elif aba == "Modelo Preditivo":
    st.subheader("🔮 Previsão de Nota de Filme")

    X = df[["budget", "revenue", "popularity", "vote_count"]]
    y = df["vote_average"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"""
    - **MSE:** {mse:.2f}  
    - **RMSE:** {rmse:.2f}  
    - **R²:** {r2:.2f}
    """)

    st.markdown("---")
    st.subheader("🎬 Faça sua própria previsão")
    budget = st.number_input("Orçamento", value=10000000)
    revenue = st.number_input("Receita", value=30000000)
    popularity = st.slider("Popularidade", 0.0, 200.0, step=1.0, value=10.0)
    vote_count = st.slider("Número de Votos", 0, 100000, step=100, value=1000)

    entrada = np.array([[budget, revenue, popularity, vote_count]])
    nota_prevista = modelo.predict(entrada)[0]

    st.success(f"🎯 Nota prevista: {nota_prevista:.2f}")
