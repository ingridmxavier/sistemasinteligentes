"""Projeto Final - Sistema de Recomendação para E-commerce.

Disciplina: Sistemas Inteligentes
Dataset: Amazon Fine Food Reviews

Objetivo
--------
Construir e avaliar um sistema de recomendação de produtos a partir do
histórico de avaliações dos usuários. O projeto utiliza filtragem colaborativa
e compara três abordagens:

1. SVD (Singular Value Decomposition), usado como modelo de recomendação;
2. KNN item-item, técnica baseada em vizinhos e similaridade entre produtos;
3. Baseline, usado como referência simples para comparação.

Etapas executadas pelo programa
-------------------------------
1. Leitura e validação do arquivo Reviews.csv;
2. Análise dos atributos e da completude do dataset;
3. Limpeza de valores ausentes, notas inválidas e duplicatas;
4. Tratamento de múltiplas avaliações do mesmo usuário para o mesmo produto;
5. Aplicação de filtro k-core em usuários e produtos;
6. Análise exploratória dos dados;
7. Separação holdout entre treinamento e teste;
8. Treinamento e avaliação do SVD;
9. Validação cruzada k-fold do SVD;
10. Comparação com Baseline e KNN item-item;
11. Avaliação por métricas de regressão, classificação e ranking Top-K;
12. Geração de gráficos, tabelas, recomendações e arquivos de resumo;
13. Treinamento e salvamento opcional de um modelo SVD final.

Execução recomendada
--------------------
Com Reviews.csv na mesma pasta deste arquivo:

    python projeto_recomendacao_ecommerce.py

Também é possível informar explicitamente o dataset:

    python projeto_recomendacao_ecommerce.py --csv "caminho/Reviews.csv"

Os resultados são armazenados, por padrão, em "resultados_projeto/".

Observações
-----------
- A semente pseudoaleatória é fixa para melhorar a reprodutibilidade.
- UserId e ProductId são identificadores categóricos e não são normalizados.
- Score já está definido na escala ordinal de 1 a 5.
- O código não utiliza dataset de fallback: a execução exige o Reviews.csv
  do projeto para evitar resultados produzidos com uma base diferente.
"""


from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

SEED = 99


# =============================================================================
# 1. CONFIGURAÇÃO, ARGUMENTOS E UTILITÁRIOS
# =============================================================================
def criar_argumentos() -> argparse.Namespace:
    """Cria e valida os argumentos de linha de comando.

        Os valores padrão correspondem à configuração principal do projeto.
        Algumas flags existem apenas para facilitar testes e experimentação."""
    parser = argparse.ArgumentParser(
        description="Sistema de recomendação para Amazon Fine Food Reviews."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Reviews.csv"),
        help=(
            "Caminho para o arquivo Reviews.csv. "
            "Por padrão, usa Reviews.csv no diretório atual."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resultados_projeto"),
        help="Pasta de saída.",
    )
    parser.add_argument("--min-user", type=int, default=5)
    parser.add_argument("--min-item", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--factors", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--ranking-users", type=int, default=300)
    parser.add_argument("--ranking-candidates", type=int, default=2000)
    parser.add_argument("--example-users", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--knn-memory-mb", type=float, default=1024.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    # ---------------------------------------------------------------------
    # RECURSOS OPCIONAIS DE DESENVOLVIMENTO
    # Não são necessários para reproduzir a execução principal. Foram mantidos
    # porque ajudam em testes e não alteram o funcionamento padrão.
    # ---------------------------------------------------------------------
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Executa uma versão reduzida para testar o ambiente.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Faz uma pequena busca de hiperparâmetros do SVD.",
    )
    parser.add_argument(
        "--skip-knn",
        action="store_true",
        help="Não executa o KNN item-item.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Não salva o modelo SVD final treinado com todos os dados.",
    )
    args = parser.parse_args()

    if args.quick:
        args.max_rows = args.max_rows or 100_000
        args.folds = min(args.folds, 3)
        args.factors = min(args.factors, 40)
        args.epochs = min(args.epochs, 10)
        args.ranking_users = min(args.ranking_users, 100)
        args.ranking_candidates = min(args.ranking_candidates, 500)

    if not 0 < args.test_size < 0.5:
        parser.error("--test-size deve estar entre 0 e 0.5.")
    if args.min_user < 2 or args.min_item < 2:
        parser.error("--min-user e --min-item devem ser pelo menos 2.")
    if args.folds < 2:
        parser.error("--folds deve ser pelo menos 2.")
    if args.top_k < 1:
        parser.error("--top-k deve ser positivo.")
    if args.ranking_candidates < args.top_k:
        parser.error("--ranking-candidates deve ser >= --top-k.")

    return args


def importar_surprise() -> SimpleNamespace:
    """Importa os componentes do scikit-surprise usados no projeto.

        O carregamento é feito aqui para que uma dependência ausente produza uma
        mensagem de instalação clara em vez de um erro pouco informativo."""
    try:
        from surprise import BaselineOnly, Dataset, KNNWithMeans, Reader, SVD, accuracy, dump
        from surprise.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
    except ImportError as exc:
        raise SystemExit(
            "A biblioteca scikit-surprise não foi encontrada.\n"
            "Instale com:\n"
            "python -m pip install pandas numpy matplotlib scikit-learn scikit-surprise"
        ) from exc

    return SimpleNamespace(
        BaselineOnly=BaselineOnly,
        Dataset=Dataset,
        KNNWithMeans=KNNWithMeans,
        Reader=Reader,
        SVD=SVD,
        accuracy=accuracy,
        dump=dump,
        GridSearchCV=GridSearchCV,
        KFold=KFold,
        cross_validate=cross_validate,
        train_test_split=train_test_split,
    )


def preparar_pastas(output: Path) -> dict[str, Path]:
    """Cria a estrutura de diretórios usada para armazenar resultados."""
    pastas = {
        "raiz": output,
        "graficos": output / "graficos",
        "tabelas": output / "tabelas",
        "modelo": output / "modelo",
    }
    for pasta in pastas.values():
        pasta.mkdir(parents=True, exist_ok=True)
    return pastas


def validar_caminho_csv(caminho_csv: Path) -> Path:
    """Valida o caminho do dataset e interrompe a execução se ele não existir.

    O projeto foi desenvolvido especificamente para o Amazon Fine Food Reviews.
    Por esse motivo, não é utilizado outro dataset como fallback.
    """
    caminho_csv = caminho_csv.expanduser()

    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_csv}\n"
            "Coloque Reviews.csv na mesma pasta do programa ou informe "
            "o caminho com --csv."
        )

    if not caminho_csv.is_file():
        raise FileNotFoundError(
            f"O caminho informado não é um arquivo: {caminho_csv}"
        )

    return caminho_csv



def conversor_json(valor):
    """Converte tipos NumPy, pandas e Path para valores serializáveis em JSON."""
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return None if not np.isfinite(valor) else float(valor)
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, set):
        return sorted(valor)
    if pd.isna(valor):
        return None
    raise TypeError(f"Tipo não serializável: {type(valor)}")


def salvar_json(dados: dict, caminho: Path) -> None:
    """Salva um dicionário em JSON usando UTF-8 e indentação legível."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2, default=conversor_json)


# =============================================================================
# 2. LEITURA, ANÁLISE E PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================
def descricao_atributo(coluna: str) -> tuple[str, str, str]:
    """Retorna tipo conceitual, significado e uso de um atributo do dataset."""
    descricoes = {
        "Id": ("identificador", "índice da avaliação", "não usado no modelo"),
        "ProductId": (
            "categórico nominal",
            "identificador do produto",
            "usado pelo recomendador",
        ),
        "UserId": (
            "categórico nominal",
            "identificador do usuário",
            "usado pelo recomendador",
        ),
        "ProfileName": (
            "categórico nominal",
            "nome público do usuário",
            "não usado no modelo",
        ),
        "HelpfulnessNumerator": (
            "numérico discreto",
            "votos de utilidade recebidos",
            "não representa preferência pelo produto",
        ),
        "HelpfulnessDenominator": (
            "numérico discreto",
            "total de votos de utilidade",
            "não representa preferência pelo produto",
        ),
        "Score": (
            "ordinal numérico",
            "avaliação de 1 a 5",
            "alvo previsto pelo recomendador",
        ),
        "Time": (
            "temporal numérico",
            "instante Unix da avaliação",
            "usado para resolver avaliações repetidas e análise temporal",
        ),
        "Summary": ("texto", "resumo da avaliação", "não usado no modelo atual"),
        "Text": ("texto", "conteúdo da avaliação", "não usado no modelo atual"),
    }
    return descricoes.get(
        coluna,
        ("outro", "atributo adicional", "não usado no modelo atual"),
    )


def criar_perfil_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói uma tabela descritiva dos atributos do dataset.

        Para cada coluna são registrados tipo, significado, quantidade de valores
        ausentes, completude e número de valores distintos."""
    linhas = []
    for coluna in df.columns:
        tipo, significado, uso = descricao_atributo(coluna)
        faltantes = int(df[coluna].isna().sum())
        linhas.append(
            {
                "atributo": coluna,
                "dtype_original": str(df[coluna].dtype),
                "tipo_conceitual": tipo,
                "significado": significado,
                "uso": uso,
                "faltantes": faltantes,
                "completude_pct": 100.0 * (1.0 - faltantes / max(len(df), 1)),
                "valores_unicos": int(df[coluna].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(linhas)


def filtrar_k_core(
    df: pd.DataFrame, min_user: int, min_item: int
) -> tuple[pd.DataFrame, list[dict]]:
    """Aplica filtro k-core iterativo em usuários e produtos.

        A remoção de um usuário pode reduzir o número de avaliações de um produto,
        e vice-versa. Por isso o filtro é repetido até que todos os elementos
        restantes atendam simultaneamente aos limites mínimos."""
    atual = df.copy()
    historico = []

    # O limite de 100 iterações é apenas uma proteção. Na prática, o filtro
    # converge quando nenhuma nova avaliação precisa ser removida.
    for iteracao in range(1, 100):
        antes = len(atual)
        cont_user = atual["UserId"].value_counts()
        cont_prod = atual["ProductId"].value_counts()
        usuarios = cont_user[cont_user >= min_user].index
        produtos = cont_prod[cont_prod >= min_item].index
        atual = atual[
            atual["UserId"].isin(usuarios) & atual["ProductId"].isin(produtos)
        ].copy()
        historico.append(
            {
                "iteracao": iteracao,
                "avaliacoes": int(len(atual)),
                "usuarios": int(atual["UserId"].nunique()),
                "produtos": int(atual["ProductId"].nunique()),
            }
        )
        if len(atual) == antes:
            break

    return atual, historico


def carregar_amazon(
    caminho_csv: Path, args: argparse.Namespace, pastas: dict[str, Path]
) -> tuple[pd.DataFrame, dict]:
    """Carrega, analisa, limpa e filtra o Amazon Fine Food Reviews.

        A função também salva arquivos que documentam completude, seleção de
        atributos e estatísticas do pré-processamento."""
    print("Carregando Amazon Reviews...")
    bruto = pd.read_csv(caminho_csv, nrows=args.max_rows, low_memory=False)

    obrigatorias = {"UserId", "ProductId", "Score"}
    ausentes = obrigatorias - set(bruto.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(ausentes)}")

    perfil = criar_perfil_atributos(bruto)
    perfil.to_csv(
        pastas["tabelas"] / "perfil_atributos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selecao = perfil[["atributo", "tipo_conceitual", "significado", "uso"]].copy()
    selecao["selecionado_no_modelo"] = selecao["atributo"].isin(
        ["UserId", "ProductId", "Score"]
    )
    selecao.to_csv(
        pastas["tabelas"] / "selecao_atributos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
    perfil_ord = perfil.sort_values("completude_pct")
    ax_comp.barh(perfil_ord["atributo"], perfil_ord["completude_pct"])
    ax_comp.set_xlim(0, 100)
    ax_comp.set_xlabel("Completude (%)")
    ax_comp.set_title("Completude dos atributos do dataset")
    ax_comp.grid(True, alpha=0.25, axis="x")
    fig_comp.tight_layout()
    fig_comp.savefig(
        pastas["graficos"] / "completude_atributos.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_comp)

    linhas_lidas = len(bruto)
    colunas_modelo = ["UserId", "ProductId", "Score"]
    if "Time" in bruto.columns:
        colunas_modelo.append("Time")
    df = bruto[colunas_modelo].copy()

    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    if "Time" in df.columns:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

    for coluna in ["UserId", "ProductId"]:
        df[coluna] = df[coluna].astype("string").str.strip()
        df.loc[df[coluna] == "", coluna] = pd.NA

    linhas_incompletas = int(
        df[["UserId", "ProductId", "Score"]].isna().any(axis=1).sum()
    )
    notas_invalidas = int(
        (df["Score"].notna() & ~df["Score"].between(1.0, 5.0)).sum()
    )

    df = df.dropna(subset=["UserId", "ProductId", "Score"])
    df = df[df["Score"].between(1.0, 5.0)].copy()

    # Remove linhas completamente repetidas.
    duplicatas_exatas = int(df.duplicated().sum())
    df = df.drop_duplicates(keep="last")

    # Um usuário pode ter avaliado o mesmo produto mais de uma vez. Para que
    # cada par usuário-produto represente uma única interação, preservamos a
    # avaliação mais recente quando o atributo Time está disponível.
    pares_repetidos = int(df.duplicated(subset=["UserId", "ProductId"]).sum())
    if pares_repetidos:
        if "Time" in df.columns:
            df["_ordem"] = np.arange(len(df))
            df = df.sort_values(["UserId", "ProductId", "Time", "_ordem"])
            df = df.drop_duplicates(subset=["UserId", "ProductId"], keep="last")
            df = df.drop(columns="_ordem")
        else:
            df = (
                df.groupby(["UserId", "ProductId"], as_index=False, sort=False)["Score"]
                .mean()
            )

    # O k-core é aplicado depois da limpeza para que os limites mínimos sejam
    # satisfeitos no conjunto que realmente será entregue aos modelos.
    antes_k_core = len(df)
    df, historico = filtrar_k_core(df, args.min_user, args.min_item)
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Nenhuma avaliação restou após o filtro. Reduza --min-user e --min-item."
        )

    print(f"Avaliações após filtro: {df.shape[0]}")

    usuarios = int(df["UserId"].nunique())
    produtos = int(df["ProductId"].nunique())
    avaliacoes = int(len(df))
    esparsidade = 1.0 - avaliacoes / max(usuarios * produtos, 1)

    resumo = {
        "arquivo": str(caminho_csv),
        "linhas_lidas": int(linhas_lidas),
        "linhas_incompletas_removidas": linhas_incompletas,
        "notas_fora_da_escala_1_5": notas_invalidas,
        "duplicatas_exatas_removidas": duplicatas_exatas,
        "pares_usuario_produto_repetidos": pares_repetidos,
        "avaliacoes_antes_k_core": int(antes_k_core),
        "avaliacoes_finais": avaliacoes,
        "usuarios_finais": usuarios,
        "produtos_finais": produtos,
        "esparsidade": float(esparsidade),
        "nota_media": float(df["Score"].mean()),
        "nota_mediana": float(df["Score"].median()),
        "nota_desvio_padrao": float(df["Score"].std(ddof=0)),
        "min_avaliacoes_usuario": int(df["UserId"].value_counts().min()),
        "max_avaliacoes_usuario": int(df["UserId"].value_counts().max()),
        "min_avaliacoes_produto": int(df["ProductId"].value_counts().min()),
        "max_avaliacoes_produto": int(df["ProductId"].value_counts().max()),
        "normalizacao_aplicada": False,
        "justificativa_escala": (
            "UserId e ProductId são identificadores categóricos e não devem ser "
            "normalizados. Score já está na mesma escala ordinal de 1 a 5."
        ),
        "atributos_usados": ["UserId", "ProductId", "Score"],
        "historico_filtro_k_core": historico,
    }

    df.to_csv(
        pastas["tabelas"] / "dados_preprocessados.csv",
        index=False,
        encoding="utf-8-sig",
    )
    salvar_json(resumo, pastas["tabelas"] / "resumo_preprocessamento.json")

    return df, resumo



def criar_dataset_surprise(df: pd.DataFrame, api: SimpleNamespace):
    """Converte o DataFrame para a representação exigida pelo Surprise."""
    reader = api.Reader(rating_scale=(1, 5))
    return api.Dataset.load_from_df(df[["UserId", "ProductId", "Score"]], reader)


def separar_dataframes_por_testset(
    df: pd.DataFrame, testset: list[tuple]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstrói DataFrames de treino e teste a partir do holdout do Surprise.

        Essa separação é usada na avaliação de ranking para garantir que as
        recomendações sejam construídas somente a partir do histórico de treino."""
    chaves_teste = {(str(uid), str(iid)) for uid, iid, _ in testset}
    chaves = list(zip(df["UserId"].astype(str), df["ProductId"].astype(str)))
    mascara = np.fromiter((chave in chaves_teste for chave in chaves), dtype=bool)
    return df.loc[~mascara].copy(), df.loc[mascara].copy()


# =============================================================================
# 3. MODELOS E MÉTRICAS DE DESEMPENHO
# =============================================================================
def parametros_svd(args: argparse.Namespace) -> dict:
    """Retorna os hiperparâmetros padrão do modelo SVD."""
    return {
        "n_factors": args.factors,
        "n_epochs": args.epochs,
        "lr_all": 0.005,
        "reg_all": 0.02,
        "random_state": SEED,
    }


def selecionar_hiperparametros(
    treino_df: pd.DataFrame,
    args: argparse.Namespace,
    api: SimpleNamespace,
    pasta_tabelas: Path,
) -> dict:
    """Seleciona parâmetros do SVD quando --tune está ativo.

        A busca utiliza somente os dados de treinamento do holdout, evitando usar
        o conjunto de teste para escolher hiperparâmetros."""
    if not args.tune:
        return parametros_svd(args)

    print("\nBuscando hiperparâmetros do SVD...")
    dados_treino = criar_dataset_surprise(treino_df, api)
    grade = {
        "n_factors": sorted({max(20, args.factors // 2), args.factors}),
        "n_epochs": sorted({max(10, args.epochs // 2), args.epochs}),
        "lr_all": [0.003, 0.005],
        "reg_all": [0.02, 0.05],
    }
    cv = api.KFold(n_splits=min(3, args.folds), random_state=SEED, shuffle=True)
    busca = api.GridSearchCV(
        api.SVD,
        grade,
        measures=["rmse", "mae"],
        cv=cv,
        n_jobs=args.n_jobs,
        joblib_verbose=0,
    )
    busca.fit(dados_treino)
    melhores = dict(busca.best_params["rmse"])
    melhores["random_state"] = SEED

    pd.DataFrame(busca.cv_results).to_csv(
        pasta_tabelas / "busca_hiperparametros_svd.csv",
        index=False,
        encoding="utf-8-sig",
    )
    salvar_json(
        {"melhores_parametros_rmse": melhores},
        pasta_tabelas / "melhores_hiperparametros_svd.json",
    )
    print(f"Melhores parâmetros: {melhores}")
    return melhores


def extrair_vetores(predicoes) -> tuple[np.ndarray, np.ndarray]:
    """Extrai vetores NumPy de ratings reais e previstos."""
    reais = np.asarray([p.r_ui for p in predicoes], dtype=float)
    previstos = np.asarray([p.est for p in predicoes], dtype=float)
    return reais, previstos


def metricas_completas(
    predicoes,
    threshold: float,
) -> dict:
    """Calcula métricas de regressão e classificação.

        Para as métricas de classificação, ratings maiores ou iguais ao limiar são
        tratados como relevantes e os demais como não relevantes."""
    reais, previstos = extrair_vetores(predicoes)
    y_real = (reais >= threshold).astype(int)
    y_pred = (previstos >= threshold).astype(int)

    rmse = float(np.sqrt(mean_squared_error(reais, previstos)))
    mae = float(np.mean(np.abs(reais - previstos)))
    mse = float(mean_squared_error(reais, previstos))
    r2 = float(r2_score(reais, previstos)) if len(reais) >= 2 else math.nan

    resultado = {
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "Acuracia": float(accuracy_score(y_real, y_pred)),
        "Acuracia_balanceada": float(balanced_accuracy_score(y_real, y_pred)),
        "Precisao": float(precision_score(y_real, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_real, y_pred, zero_division=0)),
        "F1": float(f1_score(y_real, y_pred, zero_division=0)),
    }
    if len(np.unique(y_real)) == 2:
        resultado["ROC_AUC"] = float(roc_auc_score(y_real, previstos))
    else:
        resultado["ROC_AUC"] = math.nan
    return resultado


def treinar_modelos_adicionais(
    api: SimpleNamespace,
    trainset,
    testset,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, list]]:
    """Treina Baseline e KNN item-item no mesmo conjunto de treinamento.

        O Baseline funciona como referência simples. O KNN item-item compara
        produtos por similaridade cosseno."""
    modelos = {}
    predicoes = {}

    print("\nTreinando baseline de média + vieses...")
    baseline = api.BaselineOnly(verbose=False)
    baseline.fit(trainset)
    modelos["Baseline"] = baseline
    predicoes["Baseline"] = baseline.test(testset)

    if args.skip_knn:
        print("KNN ignorado por --skip-knn.")
        return modelos, predicoes

    # O KNN item-item mantém uma matriz de similaridade entre produtos.
    # Como essa matriz cresce aproximadamente com o quadrado do número de itens,
    # estimamos o uso de memória antes do treinamento.
    n_itens = trainset.n_items
    memoria_estimada = (n_itens * n_itens * 8) / (1024**2)
    if memoria_estimada > args.knn_memory_mb:
        print(
            "KNN não executado: matriz de similaridade estimada em "
            f"{memoria_estimada:.1f} MB, acima do limite de "
            f"{args.knn_memory_mb:.1f} MB. Use --knn-memory-mb para alterar."
        )
        return modelos, predicoes

    print("Treinando KNN item-item...")
    knn = api.KNNWithMeans(
        k=40,
        min_k=1,
        sim_options={"name": "cosine", "user_based": False},
        verbose=False,
    )
    knn.fit(trainset)
    modelos["KNN item-item"] = knn
    predicoes["KNN item-item"] = knn.test(testset)
    return modelos, predicoes


# =============================================================================
# 4. AVALIAÇÃO DE RANKING E GERAÇÃO DE RECOMENDAÇÕES
# =============================================================================
def top_n_recomendacoes(predicoes, n=5):
    """Retorna as N maiores predições do testset para cada usuário.

        Esta função preserva uma saída da versão inicial. Ela serve como exemplo de
        predições no conjunto de teste e não deve ser confundida com a recomendação
        sobre itens ainda não avaliados, feita posteriormente."""
    top_n = defaultdict(list)
    for uid, iid, _, est, _ in predicoes:
        top_n[uid].append((iid, est))
    for uid, avals in top_n.items():
        avals.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = avals[:n]
    return top_n


def dcg_binario(relevancias: list[int]) -> float:
    """Calcula o Discounted Cumulative Gain para relevâncias binárias."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevancias))


def avaliar_ranking(
    modelo,
    treino_df: pd.DataFrame,
    teste_df: pd.DataFrame,
    k: int,
    threshold: float,
    max_users: int,
    max_candidates: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Avalia recomendações Top-K.

        Calcula Precision@K, Recall@K, HitRate@K, NDCG@K, MAP@K e Coverage.
        Quando o catálogo é grande, candidatos negativos podem ser amostrados para
        manter o custo computacional controlado."""
    rng = np.random.default_rng(seed)
    catalogo = np.asarray(sorted(treino_df["ProductId"].astype(str).unique()), dtype=object)
    vistos_por_usuario = treino_df.groupby("UserId")["ProductId"].apply(
        lambda s: set(s.astype(str))
    )

    positivos = teste_df[teste_df["Score"] >= threshold].copy()
    positivos["UserId"] = positivos["UserId"].astype(str)
    positivos["ProductId"] = positivos["ProductId"].astype(str)
    relevantes_por_usuario = positivos.groupby("UserId")["ProductId"].apply(set)

    usuarios = [u for u in relevantes_por_usuario.index if u in vistos_por_usuario.index]
    if len(usuarios) > max_users:
        usuarios = list(rng.choice(np.asarray(usuarios, dtype=object), max_users, replace=False))

    linhas = []
    recomendados_globais = set()
    protocolo_amostrado = False

    for uid in usuarios:
        vistos = vistos_por_usuario[uid]
        relevantes = set(relevantes_por_usuario[uid]) - vistos
        if not relevantes:
            continue

        candidatos = [item for item in catalogo if item not in vistos]
        if not candidatos:
            continue

        # Para catálogos grandes, preservamos todos os itens relevantes e
        # amostramos apenas parte dos itens negativos. Isso reduz o custo da
        # avaliação Top-K sem eliminar os positivos que precisam ser encontrados.
        if len(candidatos) > max_candidates:
            protocolo_amostrado = True
            negativos = [item for item in candidatos if item not in relevantes]
            limite_negativos = max(max_candidates - len(relevantes), k)
            if len(negativos) > limite_negativos:
                negativos = list(
                    rng.choice(np.asarray(negativos, dtype=object), limite_negativos, replace=False)
                )
            candidatos = list(relevantes) + negativos

        pontuacoes = [(iid, modelo.predict(uid, iid).est) for iid in candidatos]
        pontuacoes.sort(key=lambda x: x[1], reverse=True)
        recomendados = [iid for iid, _ in pontuacoes[:k]]
        recomendados_globais.update(recomendados)

        rel_top = [1 if iid in relevantes else 0 for iid in recomendados]
        hits = sum(rel_top)
        precision_k = hits / k
        recall_k = hits / len(relevantes)
        hit_rate = float(hits > 0)
        dcg = dcg_binario(rel_top)
        ideal = dcg_binario([1] * min(len(relevantes), k))
        ndcg = dcg / ideal if ideal > 0 else 0.0

        acumulado = 0.0
        hits_parciais = 0
        for posicao, rel in enumerate(rel_top, start=1):
            if rel:
                hits_parciais += 1
                acumulado += hits_parciais / posicao
        ap_k = acumulado / min(len(relevantes), k) if relevantes else 0.0

        linhas.append(
            {
                "UserId": uid,
                "relevantes_teste": len(relevantes),
                "hits": hits,
                f"Precision@{k}": precision_k,
                f"Recall@{k}": recall_k,
                f"HitRate@{k}": hit_rate,
                f"NDCG@{k}": ndcg,
                f"MAP@{k}": ap_k,
                "candidatos_avaliados": len(candidatos),
            }
        )

    detalhes = pd.DataFrame(linhas)
    if detalhes.empty:
        return {
            "usuarios_avaliados": 0,
            f"Precision@{k}": math.nan,
            f"Recall@{k}": math.nan,
            f"HitRate@{k}": math.nan,
            f"NDCG@{k}": math.nan,
            f"MAP@{k}": math.nan,
            "Coverage": math.nan,
            "protocolo": "sem usuários elegíveis",
        }, detalhes

    metricas = {
        "usuarios_avaliados": int(len(detalhes)),
        f"Precision@{k}": float(detalhes[f"Precision@{k}"].mean()),
        f"Recall@{k}": float(detalhes[f"Recall@{k}"].mean()),
        f"HitRate@{k}": float(detalhes[f"HitRate@{k}"].mean()),
        f"NDCG@{k}": float(detalhes[f"NDCG@{k}"].mean()),
        f"MAP@{k}": float(detalhes[f"MAP@{k}"].mean()),
        "Coverage": float(len(recomendados_globais) / max(len(catalogo), 1)),
        "protocolo": (
            f"amostrado, até {max_candidates} candidatos por usuário"
            if protocolo_amostrado
            else "catálogo completo"
        ),
    }
    return metricas, detalhes


def recomendar_catalogo(
    modelo,
    treino_df: pd.DataFrame,
    usuarios: list[str],
    n: int,
) -> pd.DataFrame:
    """Gera recomendações entre produtos ainda não avaliados pelo usuário."""
    catalogo = sorted(treino_df["ProductId"].astype(str).unique())
    vistos = treino_df.groupby("UserId")["ProductId"].apply(lambda s: set(s.astype(str)))
    linhas = []

    for uid in usuarios:
        if uid not in vistos.index:
            continue
        candidatos = [iid for iid in catalogo if iid not in vistos[uid]]
        pontuacoes = [(iid, modelo.predict(uid, iid).est) for iid in candidatos]
        pontuacoes.sort(key=lambda x: x[1], reverse=True)
        for rank, (iid, score) in enumerate(pontuacoes[:n], start=1):
            linhas.append(
                {
                    "UserId": uid,
                    "Rank": rank,
                    "ProductId": iid,
                    "Score_previsto": score,
                }
            )
    return pd.DataFrame(linhas)


# =============================================================================
# 5. FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================
def salvar_figura(fig, caminho: Path) -> None:
    """Aplica layout, salva uma figura em PNG e libera sua memória."""
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grafico_analise_exploratoria(df: pd.DataFrame, caminho: Path) -> None:
    """Gera gráficos sobre notas e quantidade de avaliações por entidade."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Análise Exploratória do Dataset", fontsize=15, fontweight="bold")

    cont_notas = df["Score"].value_counts().sort_index()
    axes[0, 0].bar(cont_notas.index.astype(str), cont_notas.values)
    axes[0, 0].set_title("Distribuição das avaliações")
    axes[0, 0].set_xlabel("Score")
    axes[0, 0].set_ylabel("Quantidade")

    cont_user = df["UserId"].value_counts()
    axes[0, 1].hist(cont_user.values, bins=40)
    axes[0, 1].set_title("Avaliações por usuário")
    axes[0, 1].set_xlabel("Quantidade de avaliações")
    axes[0, 1].set_ylabel("Usuários")
    axes[0, 1].set_yscale("log")

    cont_prod = df["ProductId"].value_counts()
    axes[1, 0].hist(cont_prod.values, bins=40)
    axes[1, 0].set_title("Avaliações por produto")
    axes[1, 0].set_xlabel("Quantidade de avaliações")
    axes[1, 0].set_ylabel("Produtos")
    axes[1, 0].set_yscale("log")

    medias_prod = df.groupby("ProductId")["Score"].mean()
    axes[1, 1].hist(medias_prod.values, bins=30)
    axes[1, 1].set_title("Média de score por produto")
    axes[1, 1].set_xlabel("Score médio")
    axes[1, 1].set_ylabel("Produtos")

    salvar_figura(fig, caminho)


def grafico_matriz_confusao(y_real, y_pred, caminho: Path, threshold: float) -> None:
    """Gera a matriz de confusão da classificação derivada dos ratings."""
    matriz = confusion_matrix(y_real, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    imagem = ax.imshow(matriz, cmap="Blues")
    fig.colorbar(imagem, ax=ax)
    ax.set_xticks([0, 1], [f"< {threshold:g}", f">= {threshold:g}"])
    ax.set_yticks([0, 1], [f"< {threshold:g}", f">= {threshold:g}"])
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    ax.set_title("Matriz de Confusão - SVD")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center")
    salvar_figura(fig, caminho)


def grafico_comparacao_modelos(tabela: pd.DataFrame, caminho: Path) -> None:
    """Compara visualmente RMSE e MAE dos modelos."""
    colunas = [c for c in ["RMSE", "MAE"] if c in tabela.columns]
    if not colunas:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tabela))
    largura = 0.35
    for indice, coluna in enumerate(colunas):
        desloc = (indice - (len(colunas) - 1) / 2) * largura
        ax.bar(x + desloc, tabela[coluna].values, width=largura, label=coluna)
    ax.set_xticks(x, tabela.index, rotation=15)
    ax.set_ylabel("Erro")
    ax.set_title("Comparação dos modelos no holdout")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    salvar_figura(fig, caminho)


def grafico_ranking(tabela: pd.DataFrame, k: int, caminho: Path) -> None:
    """Compara visualmente as principais métricas Top-K."""
    metricas = [f"Precision@{k}", f"Recall@{k}", f"HitRate@{k}", f"NDCG@{k}", f"MAP@{k}"]
    metricas = [m for m in metricas if m in tabela.columns]
    if tabela.empty or not metricas:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(metricas))
    largura = 0.8 / max(len(tabela), 1)
    for i, (modelo, linha) in enumerate(tabela.iterrows()):
        valores = [linha[m] for m in metricas]
        ax.bar(x + i * largura, valores, largura, label=modelo)
    ax.set_xticks(x + largura * (len(tabela) - 1) / 2, metricas)
    ax.set_ylim(0, 1)
    ax.set_title(f"Métricas de ranking Top-{k}")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    salvar_figura(fig, caminho)


# =============================================================================
# 6. SAÍDAS GRÁFICAS MANTIDAS DA VERSÃO INICIAL
# =============================================================================
# Estas saídas foram preservadas para não alterar resultados já utilizados
# durante o desenvolvimento do projeto.
def gerar_saidas_originais(
    resultados_cv: dict,
    ratings_reais: list[float],
    ratings_previstos: list[float],
    top_n,
    trainset_full,
    sparsidade: float,
    rmse: float,
    mae: float,
    r2: float,
    acuracia: float,
    precisao: float,
    recall: float,
    pasta_saida: Path,
) -> None:
    """Gera as figuras e tabelas presentes na versão inicial do projeto.

        Essas saídas foram mantidas para preservar compatibilidade com os resultados
        já utilizados pelo grupo durante o desenvolvimento."""
    rng = np.random.default_rng(SEED)
    amostra = rng.choice(len(ratings_reais), min(500, len(ratings_reais)), replace=False)
    reais_amostra = [ratings_reais[i] for i in amostra]
    previstos_amostra = [ratings_previstos[i] for i in amostra]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sistema de Recomendação - Resultados", fontsize=14, fontweight="bold")

    folds = range(1, len(resultados_cv["test_rmse"]) + 1)
    axes[0].plot(folds, resultados_cv["test_rmse"], "o-", color="#e74c3c", label="RMSE", linewidth=2)
    axes[0].plot(folds, resultados_cv["test_mae"], "s-", color="#3498db", label="MAE", linewidth=2)
    axes[0].set_title("Métricas por Fold")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Erro")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(ratings_reais, bins=20, alpha=0.6, color="#2ecc71", label="Real", edgecolor="white")
    axes[1].hist(ratings_previstos, bins=20, alpha=0.6, color="#e67e22", label="Previsto", edgecolor="white")
    axes[1].set_title("Distribuição de Ratings")
    axes[1].set_xlabel("Rating")
    axes[1].set_ylabel("Frequência")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(reais_amostra, previstos_amostra, alpha=0.3, color="#9b59b6", s=15)
    axes[2].plot([1, 5], [1, 5], "r--", linewidth=2, label="Predição perfeita")
    axes[2].set_title("Rating Real vs Previsto")
    axes[2].set_xlabel("Rating Real")
    axes[2].set_ylabel("Rating Previsto")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    salvar_figura(fig, pasta_saida / "resultados_recomendacao.png")
    print("PNG salvo: resultados_recomendacao.png")

    fig_t1, ax_t1 = plt.subplots(figsize=(7, 4))
    ax_t1.axis("off")
    ax_t1.text(
        0.5,
        0.97,
        "Métricas por Fold (Cross-Validation)",
        transform=ax_t1.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    dados_fold = [
        [
            f"Fold {i + 1}",
            f"{resultados_cv['test_rmse'][i]:.4f}",
            f"{resultados_cv['test_mae'][i]:.4f}",
        ]
        for i in range(len(resultados_cv["test_rmse"]))
    ]
    dados_fold.append(
        [
            "Média",
            f"{resultados_cv['test_rmse'].mean():.4f}",
            f"{resultados_cv['test_mae'].mean():.4f}",
        ]
    )
    dados_fold.append(
        [
            "Desvio Padrão",
            f"{resultados_cv['test_rmse'].std():.4f}",
            f"{resultados_cv['test_mae'].std():.4f}",
        ]
    )
    tab1 = ax_t1.table(
        cellText=dados_fold,
        colLabels=["Fold", "RMSE", "MAE"],
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.05, 0.9, 0.85],
    )
    tab1.auto_set_font_size(False)
    tab1.set_fontsize(11)
    for (row, col), cell in tab1.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row in [len(dados_fold) - 1, len(dados_fold)]:
            cell.set_facecolor("#dce8f5")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f4f4")
        else:
            cell.set_facecolor("white")
    salvar_figura(fig_t1, pasta_saida / "tabela_metricas.png")
    print("PNG salvo: tabela_metricas.png")

    dados_rec = []
    for usuario, recs in list(top_n.items())[:5]:
        for rank, (item, score) in enumerate(recs, 1):
            dados_rec.append(
                [str(usuario) if rank == 1 else "", str(rank), str(item), f"{score:.2f}"]
            )

    fig_t2, ax_t2 = plt.subplots(figsize=(8, 7))
    ax_t2.axis("off")
    ax_t2.text(
        0.5,
        0.97,
        "Recomendações Top-5 por Usuário",
        transform=ax_t2.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    tab2 = ax_t2.table(
        cellText=dados_rec,
        colLabels=["Usuário", "Rank", "Item", "Rating Previsto"],
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.05, 0.9, 0.85],
    )
    tab2.auto_set_font_size(False)
    tab2.set_fontsize(11)
    for (row, col), cell in tab2.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f4f4")
        else:
            cell.set_facecolor("white")
    salvar_figura(fig_t2, pasta_saida / "tabela_recomendacoes.png")
    print("PNG salvo: tabela_recomendacoes.png")

    dados_metricas = [
        ["RMSE", f"{rmse:.4f}", "Erro quadrático médio — menor é melhor"],
        ["MAE", f"{mae:.4f}", "Erro absoluto médio — menor é melhor"],
        ["R2 Score", f"{r2:.4f}", "Variação explicada — maior é melhor"],
        ["Acurácia", f"{acuracia:.4f}", "% de previsões corretas"],
        ["Precisão", f"{precisao:.4f}", "Dos recomendados, quantos o usuário gostaria"],
        ["Recall", f"{recall:.4f}", "Dos que gostaria, quantos foram recomendados"],
    ]
    fig_m, ax_m = plt.subplots(figsize=(11, 4))
    ax_m.axis("off")
    ax_m.text(
        0.5,
        0.97,
        "Todas as Métricas",
        transform=ax_m.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    tab_m = ax_m.table(
        cellText=dados_metricas,
        colLabels=["Métrica", "Valor", "Interpretação"],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.88],
    )
    tab_m.auto_set_font_size(False)
    tab_m.set_fontsize(11)
    tab_m.auto_set_column_width([0, 1, 2])
    for (row, col), cell in tab_m.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row in [1, 2, 3]:
            cell.set_facecolor("#eaf4fb")
        else:
            cell.set_facecolor("#eafaf1")
    salvar_figura(fig_m, pasta_saida / "tabela_todas_metricas.png")
    print("PNG salvo: tabela_todas_metricas.png")

    dados_resumo = [
        ["Modelo", "SVD (Filtragem Colaborativa)"],
        ["Usuários", str(trainset_full.n_users)],
        ["Itens", str(trainset_full.n_items)],
        ["Avaliações", str(trainset_full.n_ratings)],
        ["Esparsidade", f"{sparsidade:.2%}"],
        ["RMSE", f"{rmse:.4f}"],
        ["MAE", f"{mae:.4f}"],
        ["R2 Score", f"{r2:.4f}"],
        ["Acurácia", f"{acuracia:.4f}"],
        ["Precisão", f"{precisao:.4f}"],
        ["Recall", f"{recall:.4f}"],
    ]
    fig_r, ax_r = plt.subplots(figsize=(7, 5))
    ax_r.axis("off")
    ax_r.text(
        0.5,
        0.97,
        "Resumo do Projeto",
        transform=ax_r.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    tab_r = ax_r.table(
        cellText=dados_resumo,
        colLabels=["Métrica", "Valor"],
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.0, 0.9, 0.92],
    )
    tab_r.auto_set_font_size(False)
    tab_r.set_fontsize(11)
    for (row, col), cell in tab_r.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f4f4")
        else:
            cell.set_facecolor("white")
    salvar_figura(fig_r, pasta_saida / "tabela_resumo.png")
    print("PNG salvo: tabela_resumo.png")

    fig_dash, axes_dash = plt.subplots(3, 2, figsize=(16, 14))
    fig_dash.patch.set_facecolor("#f0f2f5")
    fig_dash.suptitle("Sistema de Recomendação — Dashboard", fontsize=16, fontweight="bold")

    for ax, titulo, valor, cor in [
        (axes_dash[0][0], "Total de Usuários", str(trainset_full.n_users), "#3498db"),
        (axes_dash[0][1], "Total de Itens", str(trainset_full.n_items), "#2ecc71"),
    ]:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor(cor)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.60, valor, transform=ax.transAxes, ha="center", va="center", fontsize=28, fontweight="bold", color=cor)
        ax.text(0.5, 0.25, titulo, transform=ax.transAxes, ha="center", va="center", fontsize=13, color="#555555")

    axes_dash[1][0].plot(folds, resultados_cv["test_rmse"], "o-", color="#e74c3c", label="RMSE", linewidth=2)
    axes_dash[1][0].plot(folds, resultados_cv["test_mae"], "s-", color="#3498db", label="MAE", linewidth=2)
    axes_dash[1][0].set_title("Métricas por Fold")
    axes_dash[1][0].legend()
    axes_dash[1][0].grid(True, alpha=0.3)

    axes_dash[1][1].hist(ratings_reais, bins=20, alpha=0.6, color="#2ecc71", label="Real", edgecolor="white")
    axes_dash[1][1].hist(ratings_previstos, bins=20, alpha=0.6, color="#e67e22", label="Previsto", edgecolor="white")
    axes_dash[1][1].set_title("Distribuição de Ratings")
    axes_dash[1][1].legend()
    axes_dash[1][1].grid(True, alpha=0.3)

    axes_dash[2][0].scatter(reais_amostra, previstos_amostra, alpha=0.3, color="#9b59b6", s=15)
    axes_dash[2][0].plot([1, 5], [1, 5], "r--", linewidth=2, label="Predição perfeita")
    axes_dash[2][0].set_title("Rating Real vs Previsto")
    axes_dash[2][0].legend()
    axes_dash[2][0].grid(True, alpha=0.3)

    ax_top = axes_dash[2][1]
    if top_n:
        primeiro_usuario = list(top_n.keys())[0]
        itens_top = [str(item) for item, _ in top_n[primeiro_usuario]]
        scores_top = [score for _, score in top_n[primeiro_usuario]]
        cores_bar = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]
        bars = ax_top.barh(itens_top[::-1], scores_top[::-1], color=cores_bar[: len(itens_top)])
        ax_top.set_title(f"Top-5 Recomendações (Usuário {primeiro_usuario})")
        ax_top.set_xlabel("Rating Previsto")
        ax_top.set_xlim(0, 5.8)
        for bar, score in zip(bars, scores_top[::-1]):
            ax_top.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center")
        ax_top.grid(True, alpha=0.3, axis="x")
    else:
        ax_top.axis("off")
        ax_top.text(0.5, 0.5, "Sem recomendações", ha="center", va="center")

    salvar_figura(fig_dash, pasta_saida / "dashboard.png")
    print("PNG salvo: dashboard.png")


# =============================================================================
# 7. PIPELINE PRINCIPAL
# =============================================================================
def executar_pipeline(args: argparse.Namespace) -> None:
    """Executa o experimento completo em ordem controlada."""
    inicio = time.time()
    random.seed(SEED)
    np.random.seed(SEED)

    api = importar_surprise()
    pastas = preparar_pastas(args.output)
    # O projeto exige explicitamente o Amazon Fine Food Reviews.
    # Caso o CSV não exista, a execução é interrompida com uma mensagem clara.
    caminho_csv = validar_caminho_csv(args.csv)
    df, resumo_dados = carregar_amazon(caminho_csv, args, pastas)
    data = criar_dataset_surprise(df, api)

    # Análise exploratória
    grafico_analise_exploratoria(df, pastas["graficos"] / "analise_exploratoria.png")

    trainset_full = data.build_full_trainset()
    sparsidade = 1 - trainset_full.n_ratings / (trainset_full.n_users * trainset_full.n_items)
    print(
        f"Usuários: {trainset_full.n_users} | Itens: {trainset_full.n_items} | "
        f"Ratings: {trainset_full.n_ratings} | Esparsidade: {sparsidade:.2%}"
    )

    # Holdout: o conjunto de teste é mantido separado do treinamento para
    # estimar o desempenho do modelo em avaliações não usadas no ajuste.
    trainset, testset = api.train_test_split(
        data, test_size=args.test_size, random_state=SEED
    )
    treino_df, teste_df = separar_dataframes_por_testset(df, testset)

    # Modelo SVD
    parametros = selecionar_hiperparametros(treino_df, args, api, pastas["tabelas"])
    modelo = api.SVD(**parametros)
    modelo.fit(trainset)
    print("Modelo treinado!")

    # IMPORTANTE: a validação cruzada usa outro objeto SVD. Assim o processo
    # de cross-validation não retreina o modelo do holdout e não provoca
    # vazamento de dados na avaliação final.
    modelo_cv = api.SVD(**parametros)
    cv = api.KFold(n_splits=args.folds, random_state=SEED, shuffle=True)
    resultados_cv = api.cross_validate(
        modelo_cv,
        data,
        measures=["RMSE", "MAE"],
        cv=cv,
        verbose=True,
        n_jobs=args.n_jobs,
    )
    print(
        f"RMSE médio: {resultados_cv['test_rmse'].mean():.4f} | "
        f"MAE médio: {resultados_cv['test_mae'].mean():.4f}"
    )

    # Avaliação no teste
    predicoes = modelo.test(testset)
    rmse = api.accuracy.rmse(predicoes)
    mae = api.accuracy.mae(predicoes)

    ratings_reais = [p.r_ui for p in predicoes]
    ratings_previstos = [p.est for p in predicoes]

    # Métricas adicionais
    r2 = r2_score(ratings_reais, ratings_previstos)
    y_real = [1 if r >= args.rating_threshold else 0 for r in ratings_reais]
    y_previsto = [1 if r >= args.rating_threshold else 0 for r in ratings_previstos]
    acuracia = accuracy_score(y_real, y_previsto)
    precisao = precision_score(y_real, y_previsto, zero_division=0)
    recall = recall_score(y_real, y_previsto, zero_division=0)

    print(
        f"R2: {r2:.4f} | Acurácia: {acuracia:.4f} | "
        f"Precisão: {precisao:.4f} | Recall: {recall:.4f}"
    )

    # Top-N original, mantido para compatibilidade com a versão inicial
    top_n = top_n_recomendacoes(predicoes)
    print("\nTop-5 recomendações:")
    for i, (usuario, recs) in enumerate(list(top_n.items())[:3]):
        print(f"  Usuário {usuario}:")
        for rank, (item, score) in enumerate(recs, 1):
            print(f"    {rank}. Item {item} — {score:.2f} ⭐")

    # Gera todos os arquivos originais
    gerar_saidas_originais(
        resultados_cv=resultados_cv,
        ratings_reais=ratings_reais,
        ratings_previstos=ratings_previstos,
        top_n=top_n,
        trainset_full=trainset_full,
        sparsidade=sparsidade,
        rmse=rmse,
        mae=mae,
        r2=r2,
        acuracia=acuracia,
        precisao=precisao,
        recall=recall,
        pasta_saida=pastas["raiz"],
    )

    # Métricas mais completas do SVD
    metricas_svd = metricas_completas(predicoes, args.rating_threshold)
    modelos_extra, predicoes_extra = treinar_modelos_adicionais(
        api, trainset, testset, args
    )

    linhas_modelos = [{"Modelo": "SVD", **metricas_svd}]
    for nome, preds in predicoes_extra.items():
        linhas_modelos.append(
            {"Modelo": nome, **metricas_completas(preds, args.rating_threshold)}
        )
    tabela_modelos = pd.DataFrame(linhas_modelos).set_index("Modelo")
    tabela_modelos.to_csv(
        pastas["tabelas"] / "comparacao_modelos.csv",
        encoding="utf-8-sig",
    )
    print("\nComparação de modelos no holdout:")
    print(tabela_modelos.round(4).to_string())
    grafico_comparacao_modelos(
        tabela_modelos, pastas["graficos"] / "comparacao_modelos.png"
    )

    # Matriz de confusão do SVD
    grafico_matriz_confusao(
        np.asarray(y_real),
        np.asarray(y_previsto),
        pastas["graficos"] / "matriz_confusao_svd.png",
        args.rating_threshold,
    )

    # As métricas Top-K avaliam diretamente a qualidade da lista recomendada,
    # complementando as métricas de erro usadas para prever ratings.
    print(f"\nAvaliando ranking Top-{args.top_k}...")
    modelos_ranking = {"SVD": modelo}
    if "KNN item-item" in modelos_extra:
        modelos_ranking["KNN item-item"] = modelos_extra["KNN item-item"]

    linhas_ranking = []
    for indice, (nome, modelo_rank) in enumerate(modelos_ranking.items()):
        metricas_rank, detalhes_rank = avaliar_ranking(
            modelo_rank,
            treino_df,
            teste_df,
            args.top_k,
            args.rating_threshold,
            args.ranking_users,
            args.ranking_candidates,
            SEED + indice,
        )
        linhas_ranking.append({"Modelo": nome, **metricas_rank})
        detalhes_rank.to_csv(
            pastas["tabelas"] / f"ranking_detalhado_{nome.lower().replace(' ', '_')}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    tabela_ranking = pd.DataFrame(linhas_ranking).set_index("Modelo")
    tabela_ranking.to_csv(
        pastas["tabelas"] / "metricas_ranking.csv",
        encoding="utf-8-sig",
    )
    print(tabela_ranking.round(4).to_string())
    grafico_ranking(
        tabela_ranking,
        args.top_k,
        pastas["graficos"] / "metricas_ranking.png",
    )

    # Esta etapa é diferente do Top-5 preservado da versão inicial: aqui o
    # sistema procura somente itens que ainda não aparecem no histórico do
    # usuário, representando uma recomendação de catálogo.
    usuarios_exemplo = list(treino_df["UserId"].astype(str).value_counts().index[: args.example_users])
    recomendacoes_catalogo = recomendar_catalogo(
        modelo, treino_df, usuarios_exemplo, n=5
    )
    recomendacoes_catalogo.to_csv(
        pastas["tabelas"] / "recomendacoes_catalogo_top5.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not recomendacoes_catalogo.empty:
        print("\nTop-5 no catálogo não avaliado (SVD):")
        for uid, grupo in recomendacoes_catalogo.groupby("UserId", sort=False):
            print(f"  Usuário {uid}:")
            for _, linha in grupo.iterrows():
                print(
                    f"    {int(linha['Rank'])}. Item {linha['ProductId']} — "
                    f"{linha['Score_previsto']:.2f} ⭐"
                )

    # Resumo final em JSON
    resumo_final = {
        "configuracao": vars(args),
        "dados": resumo_dados,
        "holdout": {
            "treino": int(len(treino_df)),
            "teste": int(len(teste_df)),
            "test_size": args.test_size,
            "seed": SEED,
        },
        "svd_parametros": parametros,
        "cross_validation": {
            "folds": args.folds,
            "rmse_medio": float(resultados_cv["test_rmse"].mean()),
            "rmse_desvio": float(resultados_cv["test_rmse"].std()),
            "mae_medio": float(resultados_cv["test_mae"].mean()),
            "mae_desvio": float(resultados_cv["test_mae"].std()),
        },
        "metricas_holdout_svd": metricas_svd,
        "tempo_total_segundos": float(time.time() - inicio),
    }
    salvar_json(resumo_final, pastas["raiz"] / "resumo_execucao.json")

    # Modelo final para uso posterior
    if not args.no_save_model:
        print("\nTreinando modelo SVD final com todos os dados...")
        modelo_final = api.SVD(**parametros)
        modelo_final.fit(trainset_full)
        api.dump.dump(str(pastas["modelo"] / "svd_final.pkl"), algo=modelo_final)
        print("Modelo salvo: modelo/svd_final.pkl")

    print("\nPipeline concluído com sucesso!")
    print(f"Resultados salvos em: {pastas['raiz'].resolve()}")


def main() -> None:
    """Ponto de entrada do programa."""
    args = criar_argumentos()
    executar_pipeline(args)


if __name__ == "__main__":
    main()