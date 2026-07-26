# Notas para revisão antes da entrega

## Removido

- Fallback automático para MovieLens 100k.
- Busca automática do Reviews.csv em caminhos específicos de computadores pessoais.

## Mantido, mas marcado como opcional

- `--quick`
- `--tune`
- `--skip-knn`
- `--no-save-model`
- `--max-rows`
- `--knn-memory-mb`
- `--n-jobs`

Esses recursos facilitam testes e execução, mas não são necessários para a
metodologia principal.

## Mantido por compatibilidade

A função que gera as imagens da versão inicial foi preservada para não perder
nenhuma saída já usada pelo grupo.

Também foi mantido o primeiro Top-5 mostrado no terminal. O código agora deixa
claro que ele é calculado sobre o testset. A recomendação mais próxima de um
cenário real é salva em `recomendacoes_catalogo_top5.csv`, pois busca itens que
o usuário ainda não avaliou.