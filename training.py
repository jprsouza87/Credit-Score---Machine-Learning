import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

print("1. Carregando a base de dados (Credit Risk Dataset)...")
df = pd.read_csv('credit_risk_dataset.csv')

print("2. Tratando dados ausentes e separando variáveis...")
# Essa base tem valores vazios em 'tempo de emprego' e 'taxa de juros'
# Vamos preencher tempo de emprego com 0 e a taxa de juros com a mediana do mercado
df['person_emp_length'] = df['person_emp_length'].fillna(0)
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# O alvo já vem como 0 (Bom Pagador) e 1 (Caloteiro)
X_bruto = df.drop('loan_status', axis=1)
y = df['loan_status']

print("3. Separando os dados em Treino (80%) e Teste (20%)...")
X_train, X_test, y_train, y_test = train_test_split(X_bruto, y, test_size=0.2, random_state=42)

print("4. Configurando o Transformador...")
# Identifica quais colunas são textos (ex: motivo do empréstimo, tipo de moradia)
colunas_categoricas = X_train.select_dtypes(include=['object']).columns.tolist()

transformador = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), colunas_categoricas)],
    remainder='passthrough'
)

X_train_tratado = transformador.fit_transform(X_train)
X_test_tratado = transformador.transform(X_test)

nomes_colunas_novas = transformador.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_tratado, columns=nomes_colunas_novas)
X_test_df = pd.DataFrame(X_test_tratado, columns=nomes_colunas_novas)

print("\nVAMOS VERIFICAR QUAL O MELHOR MODELO (CRITÉRIO: RECALL)\n")

modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

melhor_modelo = None
melhor_recall = 0 
melhor_acuracia = 0
nome_melhor_modelo = ""
resultados_modelos = []

for nome, algoritmo in modelos.items():
    print(f"Treinando {nome}...")
    
    algoritmo.fit(X_train_df, y_train)
    previsoes = algoritmo.predict(X_test_df)
    
    acuracia = accuracy_score(y_test, previsoes)
    # Foco total em achar a classe 1 (Inadimplência)
    recall_inadimplentes = recall_score(y_test, previsoes, pos_label=1)
    
    print(f"Acurácia Geral: {acuracia * 100:.2f}%")
    print(f" RECALL (Detecção de Mau Pagador): {recall_inadimplentes * 100:.2f}%\n")

    resultados_modelos.append(
        {
            "modelo": nome,
            "acuracia": acuracia * 100,
            "recall": recall_inadimplentes * 100,
        }
    )
    
    if recall_inadimplentes > melhor_recall:
        melhor_recall = recall_inadimplentes
        melhor_acuracia = acuracia
        melhor_modelo = algoritmo
        nome_melhor_modelo = nome

print(f"O MELHOR MODELO (Melhor Recall) FOI: {nome_melhor_modelo}")
print(f"Acurácia do campeão: {melhor_acuracia * 100:.1f}%")
print(f"Recall do campeão:   {melhor_recall * 100:.1f}%")
print(f" Conseguiu barrar {melhor_recall * 100:.1f}% dos verdadeiros Inadimplentes!")

print("\n5. Gerando gráfico comparativo de recall entre todos os modelos testados...")
resultados_df = pd.DataFrame(resultados_modelos).sort_values("recall", ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
cores = ["#2ca02c" if nome == nome_melhor_modelo else "#1f77b4" for nome in resultados_df["modelo"]]
ax.bar(resultados_df["modelo"], resultados_df["recall"], color=cores)
ax.set_ylim(0, 100)
ax.set_ylabel("Recall (%)")
ax.set_title("Comparativo de Recall entre os Modelos Testados")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.tick_params(axis="x", rotation=20)

for i, valor in enumerate(resultados_df["recall"]):
    ax.text(i, valor + 1.5, f"{valor:.1f}%", ha="center", va="bottom", fontsize=10)

plt.tight_layout()

saida_grafico = Path("resultados")
saida_grafico.mkdir(parents=True, exist_ok=True)
caminho_imagem = saida_grafico / "comparativo_recall_modelos.png"
plt.savefig(caminho_imagem, dpi=150, bbox_inches="tight")
print(f"✅ Gráfico salvo em: {caminho_imagem}")

plt.show()

print("\n6. Gerando gráfico comparativo de acurácia entre todos os modelos testados...")
resultados_acuracia = resultados_df.sort_values("acuracia", ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
cores = ["#9467bd" if nome == nome_melhor_modelo else "#ff7f0e" for nome in resultados_acuracia["modelo"]]
ax.bar(resultados_acuracia["modelo"], resultados_acuracia["acuracia"], color=cores)
ax.set_ylim(0, 100)
ax.set_ylabel("Acurácia (%)")
ax.set_title("Comparativo de Acurácia entre os Modelos Testados")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.tick_params(axis="x", rotation=20)

for i, valor in enumerate(resultados_acuracia["acuracia"]):
    ax.text(i, valor + 1.5, f"{valor:.1f}%", ha="center", va="bottom", fontsize=10)

plt.tight_layout()

caminho_imagem_acuracia = saida_grafico / "comparativo_acuracia_modelos.png"
plt.savefig(caminho_imagem_acuracia, dpi=150, bbox_inches="tight")
print(f"✅ Gráfico salvo em: {caminho_imagem_acuracia}")

plt.show()

print("\n7. Salvando o Melhor Modelo e o Transformador para o Streamlit...")
joblib.dump(melhor_modelo, 'modelo_score.pkl')
joblib.dump(transformador, 'transformador_dados.pkl')
joblib.dump(X_bruto.iloc[0:0], 'template_colunas.pkl')

print("✅ Sucesso! Os arquivos estão prontos para a interface web.")
