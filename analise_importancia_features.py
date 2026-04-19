import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split


class ModeloComTransformacao(BaseEstimator, ClassifierMixin):
    def __init__(self, transformador, modelo):
        self.transformador = transformador
        self.modelo = modelo

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X_tratado = self.transformador.transform(X)
        X_df = pd.DataFrame(X_tratado, columns=self.transformador.get_feature_names_out())
        return self.modelo.predict(X_df)


def carregar_dados(caminho_csv: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(caminho_csv)
    df["person_emp_length"] = df["person_emp_length"].fillna(0)
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    return X, y


def calcular_importancia(n_repeticoes: int) -> tuple[pd.DataFrame, float, float]:
    modelo = joblib.load("modelo_score.pkl")
    transformador = joblib.load("transformador_dados.pkl")

    X, y = carregar_dados(Path("credit_risk_dataset.csv"))
    _, X_teste, _, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    wrapper = ModeloComTransformacao(transformador, modelo)

    previsoes_base = wrapper.predict(X_teste)
    acuracia_base = accuracy_score(y_teste, previsoes_base)
    recall_base = recall_score(y_teste, previsoes_base)

    resultado = permutation_importance(
        wrapper,
        X_teste,
        y_teste,
        n_repeats=n_repeticoes,
        random_state=42,
        scoring="recall",
    )

    importancias = pd.DataFrame(
        {
            "feature": X_teste.columns,
            "queda_media_recall": resultado.importances_mean,
            "desvio_padrao": resultado.importances_std,
        }
    ).sort_values("queda_media_recall", ascending=False)

    return importancias, acuracia_base, recall_base


def exibir_grafico(importancias: pd.DataFrame, caminho_saida: Path | None = None) -> None:
    importancias_plot = importancias.sort_values("queda_media_recall", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    cores = ["#d9534f" if valor < 0 else "#1f77b4" for valor in importancias_plot["queda_media_recall"]]
    ax.barh(importancias_plot["feature"], importancias_plot["queda_media_recall"], color=cores)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Queda média de recall")
    ax.set_ylabel("Feature")
    ax.set_title("Importância das Features na Decisão do Modelo")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if caminho_saida is not None:
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {caminho_saida}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mede a importância das features do modelo de crédito via permutation importance."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Quantidade de features exibidas no console.",
    )
    parser.add_argument(
        "--repeticoes",
        type=int,
        default=10,
        help="Número de repetições da permutação.",
    )
    parser.add_argument(
        "--saida-csv",
        type=Path,
        default=None,
        help="Salva o ranking completo em CSV.",
    )
    parser.add_argument(
        "--saida-grafico",
        type=Path,
        default=Path("resultados") / "importancia_features.png",
        help="Salva o gráfico em arquivo.",
    )
    args = parser.parse_args()

    importancias, acuracia_base, recall_base = calcular_importancia(args.repeticoes)

    print(f"Acurácia base do modelo: {acuracia_base * 100:.2f}%")
    print(f"Recall base do modelo:   {recall_base * 100:.2f}%")
    print()
    print(f"Top {min(args.top_n, len(importancias))} features mais importantes:")
    for _, linha in importancias.head(args.top_n).iterrows():
        print(
            f"- {linha['feature']}: "
            f"{linha['queda_media_recall']:.4f} "
            f"+/- {linha['desvio_padrao']:.4f}"
        )

    if args.saida_csv is not None:
        args.saida_csv.parent.mkdir(parents=True, exist_ok=True)
        importancias.to_csv(args.saida_csv, index=False)
        print()
        print(f"Ranking completo salvo em: {args.saida_csv}")

    exibir_grafico(importancias, args.saida_grafico)


if __name__ == "__main__":
    main()
