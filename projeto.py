import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, train_test_split
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score

# Dataset
caminho_csv = os.path.expanduser('~/Downloads/SI/dataset/Reviews.csv')

if os.path.exists(caminho_csv):
    print("Carregando Amazon Reviews...")
    df = pd.read_csv(caminho_csv)
    df = df[['UserId', 'ProductId', 'Score']].dropna()

    min_aval = 5
    cont_user = df['UserId'].value_counts()
    cont_prod = df['ProductId'].value_counts()
    df = df[df['UserId'].isin(cont_user[cont_user >= min_aval].index)]
    df = df[df['ProductId'].isin(cont_prod[cont_prod >= min_aval].index)]
    print(f"Avaliações após filtro: {df.shape[0]}")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Score']], reader)
else:
    print("Reviews.csv não encontrado, usando MovieLens 100k...")
    data = Dataset.load_builtin('ml-100k')

# Análise exploratória
trainset_full = data.build_full_trainset()
sparsidade = 1 - trainset_full.n_ratings / (trainset_full.n_users * trainset_full.n_items)
print(f"Usuários: {trainset_full.n_users} | Itens: {trainset_full.n_items} | Ratings: {trainset_full.n_ratings} | Esparsidade: {sparsidade:.2%}")

# Divisão treino/teste
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Modelo SVD
modelo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
modelo.fit(trainset)
print("Modelo treinado!")

# Cross-validation
resultados_cv = cross_validate(modelo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(f"RMSE médio: {resultados_cv['test_rmse'].mean():.4f} | MAE médio: {resultados_cv['test_mae'].mean():.4f}")

# Avaliação no teste
predicoes = modelo.test(testset)
rmse = accuracy.rmse(predicoes)
mae  = accuracy.mae(predicoes)

ratings_reais     = [p.r_ui for p in predicoes]
ratings_previstos = [p.est  for p in predicoes]

# Métricas adicionais
r2 = r2_score(ratings_reais, ratings_previstos)
y_real     = [1 if r >= 4.0 else 0 for r in ratings_reais]
y_previsto = [1 if r >= 4.0 else 0 for r in ratings_previstos]
acuracia = accuracy_score(y_real, y_previsto)
precisao = precision_score(y_real, y_previsto, zero_division=0)
recall   = recall_score(y_real, y_previsto, zero_division=0)

print(f"R2: {r2:.4f} | Acurácia: {acuracia:.4f} | Precisão: {precisao:.4f} | Recall: {recall:.4f}")

# Top-N recomendações
def top_n_recomendacoes(predicoes, n=5):
    top_n = defaultdict(list)
    for uid, iid, _, est, _ in predicoes:
        top_n[uid].append((iid, est))
    for uid, avals in top_n.items():
        avals.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = avals[:n]
    return top_n

top_n = top_n_recomendacoes(predicoes)

print("\nTop-5 recomendações:")
for i, (usuario, recs) in enumerate(list(top_n.items())[:3]):
    print(f"  Usuário {usuario}:")
    for rank, (item, score) in enumerate(recs, 1):
        print(f"    {rank}. Item {item} — {score:.2f} ⭐")

# Amostra para scatter
amostra = np.random.choice(len(ratings_reais), min(500, len(ratings_reais)), replace=False)
reais_amostra     = [ratings_reais[i]     for i in amostra]
previstos_amostra = [ratings_previstos[i] for i in amostra]

# PNG: gráficos
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Sistema de Recomendação - Resultados', fontsize=14, fontweight='bold')

folds = range(1, 6)
axes[0].plot(folds, resultados_cv['test_rmse'], 'o-', color='#e74c3c', label='RMSE', linewidth=2)
axes[0].plot(folds, resultados_cv['test_mae'],  's-', color='#3498db', label='MAE',  linewidth=2)
axes[0].set_title('Métricas por Fold')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('Erro')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(ratings_reais,     bins=20, alpha=0.6, color='#2ecc71', label='Real',    edgecolor='white')
axes[1].hist(ratings_previstos, bins=20, alpha=0.6, color='#e67e22', label='Previsto', edgecolor='white')
axes[1].set_title('Distribuição de Ratings')
axes[1].set_xlabel('Rating')
axes[1].set_ylabel('Frequência')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].scatter(reais_amostra, previstos_amostra, alpha=0.3, color='#9b59b6', s=15)
axes[2].plot([1, 5], [1, 5], 'r--', linewidth=2, label='Predição perfeita')
axes[2].set_title('Rating Real vs Previsto')
axes[2].set_xlabel('Rating Real')
axes[2].set_ylabel('Rating Previsto')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ingrid/Downloads/SI/resultados_recomendacao.png', dpi=150, bbox_inches='tight')
plt.close()
print("PNG salvo: resultados_recomendacao.png")

# PNG: tabela métricas por fold
fig_t1, ax_t1 = plt.subplots(figsize=(7, 4))
fig_t1.patch.set_facecolor('white')
ax_t1.axis('off')
ax_t1.text(0.5, 0.97, 'Métricas por Fold (Cross-Validation)',
           transform=ax_t1.transAxes, ha='center', va='top', fontsize=13, fontweight='bold')

dados_fold = [[f'Fold {i+1}', f"{resultados_cv['test_rmse'][i]:.4f}", f"{resultados_cv['test_mae'][i]:.4f}"] for i in range(5)]
dados_fold.append(['Média',        f"{resultados_cv['test_rmse'].mean():.4f}", f"{resultados_cv['test_mae'].mean():.4f}"])
dados_fold.append(['Desvio Padrão',f"{resultados_cv['test_rmse'].std():.4f}",  f"{resultados_cv['test_mae'].std():.4f}"])

tab1 = ax_t1.table(cellText=dados_fold, colLabels=['Fold', 'RMSE', 'MAE'],
                   cellLoc='center', loc='center', bbox=[0.05, 0.05, 0.9, 0.85])
tab1.auto_set_font_size(False)
tab1.set_fontsize(11)
for (row, col), cell in tab1.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row in [6, 7]:
        cell.set_facecolor('#dce8f5')
    elif row % 2 == 0:
        cell.set_facecolor('#f4f4f4')
    else:
        cell.set_facecolor('white')
plt.tight_layout()
plt.savefig('/home/ingrid/Downloads/SI/tabela_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("PNG salvo: tabela_metricas.png")

# PNG: tabela recomendações
dados_rec = []
for usuario, recs in list(top_n.items())[:5]:
    for rank, (item, score) in enumerate(recs, 1):
        dados_rec.append([str(usuario) if rank == 1 else '', str(rank), str(item), f'{score:.2f}'])

fig_t2, ax_t2 = plt.subplots(figsize=(8, 7))
fig_t2.patch.set_facecolor('white')
ax_t2.axis('off')
ax_t2.text(0.5, 0.97, 'Recomendações Top-5 por Usuário',
           transform=ax_t2.transAxes, ha='center', va='top', fontsize=13, fontweight='bold')
tab2 = ax_t2.table(cellText=dados_rec, colLabels=['Usuário', 'Rank', 'Item', 'Rating Previsto'],
                   cellLoc='center', loc='center', bbox=[0.05, 0.05, 0.9, 0.85])
tab2.auto_set_font_size(False)
tab2.set_fontsize(11)
for (row, col), cell in tab2.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f4f4f4')
    else:
        cell.set_facecolor('white')
plt.tight_layout()
plt.savefig('/home/ingrid/Downloads/SI/tabela_recomendacoes.png', dpi=150, bbox_inches='tight')
plt.close()
print("PNG salvo: tabela_recomendacoes.png")

# PNG: tabela todas as métricas
dados_metricas = [
    ['RMSE',      f'{rmse:.4f}',     'Erro quadrático médio — menor é melhor'],
    ['MAE',       f'{mae:.4f}',      'Erro absoluto médio — menor é melhor'],
    ['R2 Score',  f'{r2:.4f}',       'Variação explicada — maior é melhor'],
    ['Acurácia',  f'{acuracia:.4f}', '% de previsões corretas'],
    ['Precisão',  f'{precisao:.4f}', 'Dos recomendados, quantos o usuário gostaria'],
    ['Recall',    f'{recall:.4f}',   'Dos que gostaria, quantos foram recomendados'],
]
fig_m, ax_m = plt.subplots(figsize=(11, 4))
fig_m.patch.set_facecolor('white')
ax_m.axis('off')
ax_m.text(0.5, 0.97, 'Todas as Métricas',
          transform=ax_m.transAxes, ha='center', va='top', fontsize=13, fontweight='bold')
tab_m = ax_m.table(cellText=dados_metricas, colLabels=['Métrica', 'Valor', 'Interpretação'],
                   cellLoc='center', loc='center', bbox=[0.0, 0.0, 1.0, 0.88])
tab_m.auto_set_font_size(False)
tab_m.set_fontsize(11)
tab_m.auto_set_column_width([0, 1, 2])
for (row, col), cell in tab_m.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row in [1, 2, 3]:
        cell.set_facecolor('#eaf4fb')
    else:
        cell.set_facecolor('#eafaf1')
plt.tight_layout()
plt.savefig('/home/ingrid/Downloads/SI/tabela_todas_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("PNG salvo: tabela_todas_metricas.png")

# PNG: tabela resumo
dados_resumo = [
    ['Modelo',      'SVD (Filtragem Colaborativa)'],
    ['Usuários',    str(trainset_full.n_users)],
    ['Itens',       str(trainset_full.n_items)],
    ['Avaliações',  str(trainset_full.n_ratings)],
    ['Esparsidade', f'{sparsidade:.2%}'],
    ['RMSE',        f'{rmse:.4f}'],
    ['MAE',         f'{mae:.4f}'],
    ['R2 Score',    f'{r2:.4f}'],
    ['Acurácia',    f'{acuracia:.4f}'],
    ['Precisão',    f'{precisao:.4f}'],
    ['Recall',      f'{recall:.4f}'],
]
fig_r, ax_r = plt.subplots(figsize=(7, 5))
fig_r.patch.set_facecolor('white')
ax_r.axis('off')
ax_r.text(0.5, 0.97, 'Resumo do Projeto',
          transform=ax_r.transAxes, ha='center', va='top', fontsize=13, fontweight='bold')
tab_r = ax_r.table(cellText=dados_resumo, colLabels=['Métrica', 'Valor'],
                   cellLoc='center', loc='center', bbox=[0.05, 0.0, 0.9, 0.92])
tab_r.auto_set_font_size(False)
tab_r.set_fontsize(11)
for (row, col), cell in tab_r.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f4f4f4')
    else:
        cell.set_facecolor('white')
plt.tight_layout()
plt.savefig('/home/ingrid/Downloads/SI/tabela_resumo.png', dpi=150, bbox_inches='tight')
plt.close()
print("PNG salvo: tabela_resumo.png")

# PNG: dashboard
fig_dash, axes_dash = plt.subplots(3, 2, figsize=(16, 14))
fig_dash.patch.set_facecolor('#f0f2f5')
fig_dash.suptitle('Sistema de Recomendação — Dashboard', fontsize=16, fontweight='bold')

for ax, titulo, valor, cor in [
    (axes_dash[0][0], 'Total de Usuários', str(trainset_full.n_users), '#3498db'),
    (axes_dash[0][1], 'Total de Itens',    str(trainset_full.n_items), '#2ecc71'),
]:
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor(cor)
        spine.set_linewidth(3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.60, valor, transform=ax.transAxes,
            ha='center', va='center', fontsize=28, fontweight='bold', color=cor)
    ax.text(0.5, 0.25, titulo, transform=ax.transAxes,
            ha='center', va='center', fontsize=13, color='#555555')

ax_fold2 = axes_dash[1][0]
ax_fold2.set_facecolor('white')
ax_fold2.plot(folds, resultados_cv['test_rmse'], 'o-', color='#e74c3c', label='RMSE', linewidth=2)
ax_fold2.plot(folds, resultados_cv['test_mae'],  's-', color='#3498db', label='MAE',  linewidth=2)
ax_fold2.set_title('Métricas por Fold', fontsize=12, fontweight='bold', pad=10)
ax_fold2.set_xlabel('Fold')
ax_fold2.set_ylabel('Erro')
ax_fold2.legend()
ax_fold2.grid(True, alpha=0.3)

ax_dist2 = axes_dash[1][1]
ax_dist2.set_facecolor('white')
ax_dist2.hist(ratings_reais,     bins=20, alpha=0.6, color='#2ecc71', label='Real',    edgecolor='white')
ax_dist2.hist(ratings_previstos, bins=20, alpha=0.6, color='#e67e22', label='Previsto', edgecolor='white')
ax_dist2.set_title('Distribuição de Ratings', fontsize=12, fontweight='bold', pad=10)
ax_dist2.set_xlabel('Rating')
ax_dist2.set_ylabel('Frequência')
ax_dist2.legend()
ax_dist2.grid(True, alpha=0.3)

ax_sc2 = axes_dash[2][0]
ax_sc2.set_facecolor('white')
ax_sc2.scatter(reais_amostra, previstos_amostra, alpha=0.3, color='#9b59b6', s=15)
ax_sc2.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Predição perfeita')
ax_sc2.set_title('Rating Real vs Previsto', fontsize=12, fontweight='bold', pad=10)
ax_sc2.set_xlabel('Rating Real')
ax_sc2.set_ylabel('Rating Previsto')
ax_sc2.legend()
ax_sc2.grid(True, alpha=0.3)

ax_top2 = axes_dash[2][1]
ax_top2.set_facecolor('white')
primeiro_usuario = list(top_n.keys())[0]
itens_top  = [str(item) for item, _ in top_n[primeiro_usuario]]
scores_top = [score for _, score in top_n[primeiro_usuario]]
cores_bar  = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']
bars = ax_top2.barh(itens_top[::-1], scores_top[::-1], color=cores_bar)
ax_top2.set_title(f'Top-5 Recomendações (Usuário {primeiro_usuario})', fontsize=12, fontweight='bold', pad=10)
ax_top2.set_xlabel('Rating Previsto')
ax_top2.set_xlim(0, 5.8)
for bar, score in zip(bars, scores_top[::-1]):
    ax_top2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{score:.2f}', va='center', fontsize=10, fontweight='bold')
ax_top2.grid(True, alpha=0.3, axis='x')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/home/ingrid/Downloads/SI/dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("PNG salvo: dashboard.png")

print("\nPipeline concluído com sucesso!")