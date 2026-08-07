# Projeto Final — Sistema de Recomendação para E-commerce

## Visão geral

Este projeto implementa um sistema de recomendação usando o dataset **Amazon Fine Food Reviews**.

Dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download

O objetivo é estimar preferências dos usuários e gerar recomendações personalizadas de produtos a partir do histórico de avaliações.

O programa compara:

- **SVD** — filtragem colaborativa por fatoração de matriz;
- **KNN item-item** — recomendação por similaridade entre produtos;
- **Baseline** — modelo simples de referência.

Além da previsão de ratings, o projeto avalia diretamente a qualidade do ranking de recomendações.

## Estrutura

```text
projeto_recomendacao_ecommerce.py
README.md
requirements.txt
Reviews.csv                       # dataset; não incluído
```

Após a execução:

```text
resultados_projeto/
├── dashboard.png
├── resultados_recomendacao.png
├── tabela_metricas.png
├── tabela_recomendacoes.png
├── tabela_todas_metricas.png
├── tabela_resumo.png
├── resumo_execucao.json
├── graficos/
├── tabelas/
└── modelo/
```

## Pré-processamento

O pipeline realiza:

1. validação das colunas obrigatórias;
2. análise de tipos e completude;
3. conversão de `Score` para numérico;
4. remoção de identificadores ausentes;
5. remoção de notas fora da escala 1–5;
6. remoção de duplicatas;
7. tratamento de múltiplas avaliações do mesmo usuário para o mesmo produto;
8. filtro **k-core iterativo**.

Por padrão:

```text
mínimo de avaliações por usuário = 5
mínimo de avaliações por produto = 5
```

`UserId` e `ProductId` não são normalizados porque são identificadores categóricos. `Score` já utiliza a mesma escala de 1 a 5 em todo o dataset.

## Validação

O projeto utiliza:

- **holdout** para separar treinamento e teste;
- **validação cruzada k-fold** para avaliar a estabilidade do SVD.

O modelo usado na validação cruzada é um objeto separado do modelo usado no holdout, evitando vazamento de dados.

## Métricas

### Regressão
- RMSE
- MAE
- MSE
- R²

### Classificação derivada do rating
Por padrão, `Score >= 4` representa um item relevante.

- acurácia
- acurácia balanceada
- precisão
- recall
- F1
- ROC-AUC
- matriz de confusão

### Ranking Top-K
- Precision@K
- Recall@K
- HitRate@K
- NDCG@K
- MAP@K
- Coverage

As métricas Top-K são particularmente importantes porque medem se itens relevantes aparecem nas primeiras posições da recomendação.

## Instalação

Recomenda-se usar um ambiente virtual.

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Execução

Com `Reviews.csv` na mesma pasta:

```powershell
python projeto_recomendacao_ecommerce.py
```

Ou:

```powershell
python projeto_recomendacao_ecommerce.py --csv "caminho\Reviews.csv"
```

## Parâmetros principais

```text
--min-user
--min-item
--test-size
--folds
--factors
--epochs
--top-k
--rating-threshold
--ranking-users
--ranking-candidates
```

## Reprodutibilidade

O código usa uma semente fixa (`SEED = 99`) nas operações aleatórias controláveis.

A configuração e os principais resultados também são salvos em `resumo_execucao.json`.

## Observação sobre os dois Top-5

O primeiro Top-5 mostrado no terminal foi preservado da versão inicial e ordena as melhores predições dentro do conjunto de teste.

Já `recomendacoes_catalogo_top5.csv` busca itens que ainda não foram avaliados pelo usuário e representa melhor um cenário real de recomendação.

## Arquivos mais úteis para o relatório

- `graficos/analise_exploratoria.png`
- `graficos/comparacao_modelos.png`
- `graficos/matriz_confusao_svd.png`
- `graficos/metricas_ranking.png`
- `tabelas/comparacao_modelos.csv`
- `tabelas/metricas_ranking.csv`
- `tabelas/perfil_atributos.csv`
- `tabelas/resumo_preprocessamento.json`
- `resumo_execucao.json`

## Comando para rodar
python3 projeto.py --csv dataset/Reviews.csv
