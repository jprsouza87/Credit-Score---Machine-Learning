<div align="center">
  <h1>🏦 Credit Scoring Predictor</h1>
  <p><strong>Previsão de Risco de Inadimplência com Machine Learning</strong></p>
</div>

---
https://creditscore-ml.streamlit.app/

---

## 📋 Sobre o Projeto
Este projeto simula uma ferramenta de mesa de crédito para instituições financeiras. O objetivo é automatizar a classificação de risco de novos clientes, reduzindo a subjetividade humana e focando na métrica de **Recall** (Revocação) — crucial para garantir que o maior número possível de potenciais inadimplentes seja identificado e barrado antes da concessão.

## 🚀 Destaques Técnicos
1. **Tratamento de Dados:** Tratamento de nulos em variáveis críticas como taxa de juros e tempo de emprego.
2. **Machine Learning Bake-off:** Comparação entre 4 algoritmos (Regressão Logística, Árvore de Decisão, Random Forest e Gradient Boosting).
3. **Foco no Negócio (Recall):** Seleção do modelo campeão baseada na capacidade de detectar inadimplentes (Recall de 77%) em vez de apenas acertos gerais (Acurácia), priorizando a segurança financeira.
4. **Pipeline de Produção:** Implementação de `ColumnTransformer` e `OneHotEncoder` para garantir que a aplicação web trate os dados exatamente como o modelo foi treinado.

---

## 🛠️ Tecnologias Utilizadas
* **Python** (Linguagem base)
* **Scikit-Learn** (Modelagem e Pré-processamento)
* **Streamlit** (Interface Web)
* **Pandas** (Manipulação de dados)
* **Joblib** (Persistência de objetos de ML)

---

## 💻 Como usar a ferramenta

O dashboard foi desenhado para ser operado por um gerente ou analista de risco:

1. **Parâmetros do Cliente:** Na barra lateral, insira os dados demográficos (idade, renda, tempo de emprego) e o histórico de crédito (se já possui restrições no nome).
2. **Dados do Empréstimo:** Informe o valor solicitado e a taxa de juros proposta para a operação.
3. **Cálculo Automático:** A ferramenta calcula instantaneamente o *comprometimento de renda* do cliente.
4. **Simulação:** Clique no botão **"Simular Risco de Crédito"**.
5. **Resultado:** - O sistema retorna "Aprovado" ou "Reprovado"

---

## 👤 Autor

João Paulo R. Souza — [github.com/jprsouza87](https://github.com/jprsouza87)
streamlit run app.py
