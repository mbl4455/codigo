# app.py
import streamlit as st
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("🎬 Análise de Filmes - TMDb")

@st.cache_data
def carregar_dados():
    # Links diretos de download via Google Drive
    url_movies = "https://drive.google.com/uc?export=download&id=17dWfqGAtdKZAR0rTCT6cv7weiIcrNycZ"
    url_credits = "https://drive.google.com/uc?export=download&id=1hQwFfz4ZXtF9UYwiH7VEwThbg207XjRL"

    movies_df = pd.read_csv(url_movies)
    credits_df = pd.read_csv(url_credits, engine='python', on_bad_lines='skip')
    df = movies_df.merge(credits_df, left_on="id", right_on="movie_id")
    df.drop("movie_id", axis=1, inplace=True)
    return df

df = carregar_dados()

# Função para tratar colunas JSON (gêneros, elenco, etc.)
def parse_json_column(df, column):
    for idx, row in df.iterrows():
        try:
            df.at[idx, column] = json.loads(row[column])
        except (TypeError, json.JSONDecodeError):
            df.at[idx, column] = []
    return df

for col in ['genres', 'keywords', 'cast', 'crew']:
    df = parse_json_column(df, col)

def main_genre(genres):
    if genres:
        return genres[0].get("name", "Unknown")
    return "Unknown"

df["main_genre"] = df["genres"].apply(main_genre)

# Interface com abas
aba = st.sidebar.radio("Navegação", ["Exploração", "Visualizações", "Modelo Preditivo"])

if aba == "Exploração":
    st.subheader("📊 Estatísticas Descritivas")
    st.dataframe(df[["budget", "revenue", "vote_average", "popularity", "vote_count"]].describe())

    st.subheader("🎥 Exemplos de Filmes para Contexto")
    st.write("Top 5 por nota média:")
    top5 = df.sort_values("vote_average", ascending=False)[["title_y", "vote_average", "vote_count"]].head(5)
    st.dataframe(top5)
    
    st.write("Top 5 por número de votos:")
    top_votes5 = df.sort_values("vote_count", ascending=False)[["title_y", "vote_average", "vote_count"]].head(5)
    st.dataframe(top_votes5)

elif aba == "Visualizações":
    st.subheader("📈 Gráficos de Distribuição e Relações")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Distribuição das Notas")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["vote_average"], kde=True, ax=ax1, color='skyblue')
        st.pyplot(fig1)

    with col2:
        st.write("Orçamento vs Receita")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x="budget", y="revenue", ax=ax2)
        st.pyplot(fig2)

    st.write("Boxplot: Nota Média por Gênero")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="main_genre", y="vote_average", ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.write("Número de Filmes por Idioma Original")
    fig4, ax4 = plt.subplots(figsize=(8, 10))
    sns.countplot(data=df, y="original_language", order=df["original_language"].value_counts().index, ax=ax4)
    st.pyplot(fig4)

elif aba == "Modelo Preditivo":
    st.subheader("🔮 Previsão de Nota de Filme")
    
    # Preparar features e target
    X = df[["budget", "revenue", "popularity", "vote_count"]].fillna(0)
    y = df["vote_average"].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.markdown(f"""
    **Métricas do Modelo:**
    - MSE: {mse:.2f}  
    - RMSE: {rmse:.2f}  
    - R²: {r2:.2f}
    """)
    
    st.markdown("---")
    st.subheader("📥 Faça sua própria previsão")
    budget = st.number_input("Orçamento", min_value=0, value=10000000, step=100000)
    revenue = st.number_input("Receita", min_value=0, value=30000000, step=100000)
    popularity = st.slider("Popularidade", min_value=0.0, max_value=float(df["popularity"].max()), value=10.0, step=0.1)
    vote_count = st.slider("Número de Votos", min_value=0, max_value=int(df["vote_count"].max()), value=1000, step=100)
    
    entrada = np.array([[budget, revenue, popularity, vote_count]])
    nota_prevista = model.predict(entrada)[0]
    
    st.success(f"🎯 Nota prevista: {nota_prevista:.2f}")
