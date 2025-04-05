import streamlit as st
import ipeadatapy as ipea
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import os

# Configurar API do Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Função para listar índices disponíveis no IPEA
@st.cache_data
def get_ipea_series():
    series = ipea.list_series()
    series = series[~series['NAME'].str.contains("INATIVA", case=False)]  # Filtrar séries inativas
    return series[['CODE', 'NAME']]

# Função para buscar dados do IPEA
@st.cache_data
def get_ipea_data(series_code):
    data = ipea.timeseries(series_code)
    data = data.iloc[:, [-1, 1]].rename(columns={data.columns[-1]: series_code, data.columns[1]: 'Date'})
    data['Date'] = pd.to_datetime(data['Date'])
    return data.set_index('Date')

# Função para gerar insights com Gemini (com cache)
@st.cache_data(ttl=86400)
def get_insights(text_prompt):
    print("Gerando insights com OpenAI...")
    print(f"Prompt: {text_prompt}")

    model = genai.GenerativeModel("gemini-2.0-flash-lite-001")
    response = model.generate_content(text_prompt)
    return response.text

# Interface Streamlit
st.title("Análise de Correlação de Índices do IPEA")

# Obter lista de índices
series_list = get_ipea_series()

# Criar um dicionário para mapear código para nome
description_map = dict(zip(series_list['CODE'], series_list['NAME']))

# Filtros pré-definidos
preset_filters = {
    "Agregado M2 x IPCA": ("BM12_M2NCN12", "PRECOS12_IPCA12"),
    "Agregado M2 x IGP-M": ("BM12_M2NCN12", "IGP12_IGPM12"),
    "Agregado M2 x INPC": ("BM12_M2NCN12", "PRECOS12_INPC12"),
    "Agregado M2 x PIB": ("BM12_M2NCN12", "BM12_PIBAC12")
}

# Seleção de filtros pré-definidos
st.sidebar.header("Configuração")
selected_preset = st.sidebar.selectbox("Selecione um filtro pré-definido", ["Personalizado"] + list(preset_filters.keys()))

if selected_preset != "Personalizado":
    index1, index2 = preset_filters[selected_preset]
else:
    index1 = st.sidebar.selectbox("Selecione o primeiro índice", series_list['CODE'], format_func=lambda x: f"{description_map[x]} ({x})")
    index2 = st.sidebar.selectbox("Selecione o segundo índice", series_list['CODE'], format_func=lambda x: f"{description_map[x]} ({x})")

# Validação para evitar seleção do mesmo índice
if index1 == index2:
    st.sidebar.error("Os dois índices selecionados devem ser diferentes. Por favor, escolha índices distintos.")
    st.stop()

# Obter dados
data1 = get_ipea_data(index1)
data2 = get_ipea_data(index2)

# Unir datasets pelo índice de data
merged_data = pd.merge(data1, data2, left_index=True, right_index=True, how='inner')

# Estilização geral dos gráficos
sns.set_style("whitegrid")

# Exibir gráfico do primeiro índice
st.subheader(f"Gráfico do Índice {description_map[index1]} ({index1})")
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(data1.index, data1[index1], label=f"{description_map[index1]} ({index1})", linewidth=2, color='royalblue')
ax1.set_xlabel("Data")
ax1.set_ylabel("Valor")
ax1.legend()
st.pyplot(fig1)

# Exibir gráfico do segundo índice
st.subheader(f"Gráfico do Índice {description_map[index2]} ({index2})")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(data2.index, data2[index2], label=f"{description_map[index2]} ({index2})", linewidth=2, color='seagreen')
ax2.set_xlabel("Data")
ax2.set_ylabel("Valor")
ax2.legend()
st.pyplot(fig2)

# Exibir gráfico de correlação
st.subheader("Gráfico de Correlação")
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=merged_data[index1], y=merged_data[index2], ax=ax3, color='darkorange', alpha=0.7)
ax3.set_xlabel(f"{description_map[index1]} ({index1})")
ax3.set_ylabel(f"{description_map[index2]} ({index2})")
st.pyplot(fig3)

# Calcular coeficiente de correlação
correlation = merged_data.corr().iloc[0, 1]
st.markdown(f"### Coeficiente de correlação: **{correlation:.2f}**")

# Gerar insights com Gemini
if st.button("Gerar Insights com IA"):
    prompt = f"A correlação entre os índices {description_map[index1]} ({index1}) e {description_map[index2]} ({index2}) foi de {correlation:.2f}. O que isso pode indicar economicamente?"
    insight = get_insights(prompt)
    st.subheader("Insights da IA")
    st.write(insight)
