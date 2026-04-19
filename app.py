import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. Configuração da Página
st.set_page_config(page_title="Simulador de Crédito", page_icon="🏦", layout="wide")

st.title("🏦 Simulador Preditivo de Risco de Crédito")
st.markdown("Preencha os dados do cliente ao lado para analisar o status do crédito usando Inteligência Artificial.")

IMPORTANCIA_VARIAVEIS = pd.DataFrame(
    [
        ("loan_percent_income", 0.1644),
        ("person_home_ownership", 0.1409),
        ("loan_intent", 0.0781),
        ("person_income", 0.0716),
        ("loan_grade", 0.0505),
        ("person_emp_length", 0.0406),
        ("loan_int_rate", 0.0102),
        ("person_age", 0.0019),
        ("cb_person_default_on_file", 0.0002),
        ("cb_person_cred_hist_length", -0.0011),
        ("loan_amnt", -0.0161),
    ],
    columns=["variavel", "queda_media_recall"],
)

ROTULOS_VARIAVEIS = {
    "loan_percent_income": "Comprometimento da Renda",
    "person_home_ownership": "Situação da Moradia",
    "loan_intent": "Motivo do Empréstimo",
    "person_income": "Renda Anual (R$)",
    "loan_grade": "Nota de Crédito Interna (Grade)",
    "person_emp_length": "Tempo de Emprego (Anos)",
    "loan_int_rate": "Taxa de Juros Anual (%)",
    "person_age": "Idade",
    "cb_person_default_on_file": "Já teve restrição no nome (Serasa/SPC)?",
    "loan_amnt": "Valor Solicitado (R$)",
}

# 2. Carregando os Arquivos do Modelo (Cérebro)
@st.cache_resource
def carregar_modelo():
    modelo = joblib.load('modelo_score.pkl')
    transformador = joblib.load('transformador_dados.pkl')
    template = joblib.load('template_colunas.pkl')
    return modelo, transformador, template

modelo, transformador, template_df = carregar_modelo()

# 3. Construindo a Barra Lateral (Inputs do Usuário)
st.sidebar.header("📊 Dados do Cliente")

# Variáveis Numéricas
idade = st.sidebar.number_input("Idade", min_value=18, max_value=100, value=30)
renda_anual = st.sidebar.number_input("Renda Anual (R$)", min_value=1000, value=50000, step=1000)
tempo_emprego = st.sidebar.number_input("Tempo de Emprego (Anos)", min_value=0, max_value=50, value=5)
historico_credito = st.sidebar.number_input("Tempo de Histórico de Crédito (Anos)", min_value=0, max_value=30, value=5)

st.sidebar.markdown("---")
st.sidebar.header("💰 Dados do Empréstimo")

valor_emprestimo = st.sidebar.number_input("Valor Solicitado (R$)", min_value=500, value=10000, step=500)
taxa_juros = st.sidebar.number_input("Taxa de Juros Anual (%)", min_value=0.0, max_value=30.0, value=10.9)

# Variáveis Categóricas (Textos)
# Usamos os mesmos nomes em inglês que a base do Kaggle usou para não quebrar o modelo
moradia = st.sidebar.selectbox("Situação da Moradia", ["RENT", "OWN", "MORTGAGE", "OTHER"], 
                               format_func=lambda x: {"RENT": "Alugado", "OWN": "Próprio", "MORTGAGE": "Financiado", "OTHER": "Outro"}[x])

motivo = st.sidebar.selectbox("Motivo do Empréstimo", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                              format_func=lambda x: {"PERSONAL": "Pessoal", "EDUCATION": "Educação", "MEDICAL": "Saúde", "VENTURE": "Negócios", "HOMEIMPROVEMENT": "Reforma", "DEBTCONSOLIDATION": "Pagamento de Dívidas"}[x])

grau_emprestimo = st.sidebar.selectbox("Nota de Crédito Interna (Grade)", ["A", "B", "C", "D", "E", "F", "G"])
inadimplencia_previa = st.sidebar.selectbox("Já teve restrição no nome (Serasa/SPC)?", ["N", "Y"],
                                            format_func=lambda x: "Sim" if x == "Y" else "Não")

# 4. Processamento ao clicar no botão
if st.sidebar.button("Simular Risco de Crédito", use_container_width=True):
    
    # Calculando o comprometimento de renda automaticamente (Valor Empréstimo / Renda Anual)
    # Evita erro de divisão por zero
    comprometimento_renda = valor_emprestimo / renda_anual if renda_anual > 0 else 0
    
    # Criando um dicionário com exatamente os mesmos nomes de colunas da base original
    dados_entrada = {
        'person_age': [idade],
        'person_income': [renda_anual],
        'person_home_ownership': [moradia],
        'person_emp_length': [tempo_emprego],
        'loan_intent': [motivo],
        'loan_grade': [grau_emprestimo],
        'loan_amnt': [valor_emprestimo],
        'loan_int_rate': [taxa_juros],
        'loan_percent_income': [comprometimento_renda],
        'cb_person_default_on_file': [inadimplencia_previa],
        'cb_person_cred_hist_length': [historico_credito]
    }
    
    # Convertendo para DataFrame para o modelo ler
    df_cliente = pd.DataFrame(dados_entrada)
    
    # O Transformador traduz os textos para números
    df_tratado = transformador.transform(df_cliente)
    df_final = pd.DataFrame(df_tratado, columns=transformador.get_feature_names_out())
    
    # Previsão binária: 0 = aprovado, 1 = não aprovado
    predicao = modelo.predict(df_final)[0]
    status_credito = "Não aprovado" if predicao == 1 else "Aprovado"
    
    # 5. Exibindo o Resultado na Tela Principal
    st.subheader("Resultado da Análise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Status do Crédito", status_credito)
        
        # Lógica visual corporativa
        if predicao == 0:
            st.success("✅ Crédito aprovado com base no modelo atual.")
        else:
            st.error("🚨 Crédito não aprovado com base no modelo atual.")
            
    with col2:
        st.write("**Resumo dos Indicadores Chave:**")
        st.write(f"- Comprometimento da Renda: **{comprometimento_renda * 100:.1f}%**")
        st.write(f"- Restrição Prévia: **{'Sim' if inadimplencia_previa == 'Y' else 'Não'}**")
        st.write(f"- Estabilidade no Emprego: **{tempo_emprego} anos**")

    st.subheader("Importância das Variáveis no Modelo")
    st.caption("Gráfico baseado na queda média de recall quando cada variável é embaralhada no conjunto de teste.")

    grafico_importancia = (
        IMPORTANCIA_VARIAVEIS[
            (IMPORTANCIA_VARIAVEIS["variavel"] != "cb_person_cred_hist_length")
            & (IMPORTANCIA_VARIAVEIS["variavel"] != "loan_amnt")
        ]
        .copy()
    )
    grafico_importancia["rotulo"] = grafico_importancia["variavel"].map(ROTULOS_VARIAVEIS)
    grafico_importancia["rotulo"] = grafico_importancia["rotulo"].fillna(grafico_importancia["variavel"])
    grafico_importancia = grafico_importancia.sort_values("queda_media_recall", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    cores = ["#d9534f" if valor < 0 else "#1f77b4" for valor in grafico_importancia["queda_media_recall"]]
    ax.barh(grafico_importancia["rotulo"], grafico_importancia["queda_media_recall"], color=cores)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Queda média de recall")
    ax.set_ylabel("Campo preenchido pelo usuário")
    ax.set_title("Impacto de cada variável na decisão do modelo")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    with st.expander("Ver valores exatos usados no gráfico"):
        tabela_importancia = IMPORTANCIA_VARIAVEIS[
            (IMPORTANCIA_VARIAVEIS["variavel"] != "cb_person_cred_hist_length")
            & (IMPORTANCIA_VARIAVEIS["variavel"] != "loan_amnt")
        ].copy()
        tabela_importancia["variavel"] = tabela_importancia["variavel"].map(ROTULOS_VARIAVEIS).fillna(tabela_importancia["variavel"])
        st.dataframe(
            tabela_importancia.sort_values("queda_media_recall", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

else:
    # Mensagem de espera antes de clicar no botão
    st.info("👈 Preencha os parâmetros na barra lateral e clique em 'Simular Risco de Crédito' para ver o status do crédito.")
