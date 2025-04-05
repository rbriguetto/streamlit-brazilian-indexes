import streamlit as st
import ipeadatapy as ipea
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os
from functools import lru_cache

# Configurar API da OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Função para listar índices disponíveis no IPEA
@st.cache_data
def get_ipea_series():
    series = ipea.list_series()
    return series[['CODE', 'NAME']]

# Função para buscar dados do IPEA
@st.cache_data
def get_ipea_data(series_code):
    data = ipea.timeseries(series_code)
    data = data.iloc[:, [-1, 0]].rename(columns={data.columns[-1]: series_code, data.columns[0]: 'Date'})
    data['Date'] = pd.to_datetime(data['Date'])
    return data.set_index('Date')

# Função para gerar insights com OpenAI (com cache)
@lru_cache(maxsize=10)
def get_insights(text_prompt, context):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": text_prompt}
        ]
    )
    return response.choices[0].message.content

# Interface Streamlit
st.title("Análise de Correlação de Índices do IPEA")

# Obter lista de índices
series_list = get_ipea_series()

# Criar um dicionário para mapear código para nome
description_map = dict(zip(series_list['CODE'], series_list['NAME']))

# Seleção de índices na barra lateral
st.sidebar.header("Configuração")
index1 = st.sidebar.selectbox("Selecione o primeiro índice", series_list['CODE'], format_func=lambda x: f"{description_map[x]} ({x})")
index2 = st.sidebar.selectbox("Selecione o segundo índice", series_list['CODE'], format_func=lambda x: f"{description_map[x]} ({x})")

if index1 and index2:
    # Obter dados
    data1 = get_ipea_data(index1)
    data2 = get_ipea_data(index2)
    
    # Unir datasets pelo índice de data
    merged_data = pd.merge(data1, data2, left_index=True, right_index=True, how='inner')
    
    # Exibir gráfico de correlação
    st.subheader("Gráfico de Correlação")
    fig, ax = plt.subplots()
    sns.scatterplot(x=merged_data[index1], y=merged_data[index2], ax=ax)
    ax.set_xlabel(f"{description_map[index1]} ({index1})")
    ax.set_ylabel(f"{description_map[index2]} ({index2})")
    st.pyplot(fig)
    
    # Calcular coeficiente de correlação
    correlation = merged_data.corr().iloc[0, 1]
    st.write(f"Coeficiente de correlação: {correlation:.2f}")
    
    # Gerar insights com OpenAI
    if st.button("Gerar Insights com IA"):
        context = f"Você é um economista especialista em análise de dados. Os índices analisados são: {description_map[index1]} ({index1}) e {description_map[index2]} ({index2})."
        prompt = f"A correlação entre os índices {description_map[index1]} ({index1}) e {description_map[index2]} ({index2}) foi de {correlation:.2f}. O que isso pode indicar economicamente?"
        insight = get_insights(prompt, context)
        st.subheader("Insights da IA")
        st.write(insight)
