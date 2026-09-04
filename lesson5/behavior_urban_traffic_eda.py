"""
Link to dataset https://www.openml.org/search?type=data&status=active&id=42896
Sao Paulo in Brazil from December 14, 2009 to December 18, 2009 (From Monday to Friday).
Registered from 7:00 to 20:00 every 30 minutes.

Analise exploratoria e multivariada deste dataframe: estatisticas descritivas
(media, mediana, moda, desvio padrao, variancia, quartis), deteccao de outliers
(boxplot + IQR), dados ausentes/duplicados, correlacao (heatmap), graficos de
dispersao e uma analise temporal (horario x dia da semana). Os graficos sao
salvos em lesson5/output/behavior_urban_traffic/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils.eda_helpers import setup_output_dir, save_figure, numeric_summary, outlier_report, missing_data_report
from utils.html_report import build_html_report

import matplotlib.pyplot as plt
import pandas
import seaborn
import sklearn.datasets as skdatasets

OUTPUT_DIR = setup_output_dir(__file__, "behavior_urban_traffic")

print("Fetching dataset...")
df: pandas.DataFrame = skdatasets.fetch_openml(data_id=42896, as_frame=True).data
print("Dataset fetched!")

print("\nestrutura e tipos\n".upper())
print(df.info())
print(df.head(10))

"""
Features:
- Hour: categorica ordinal (7:00 a 20:00, a cada 30 min). 27 categorias.
- 16 colunas de incidentes (Immobilized_bus ... Intermittent_Semaphore): contagem
  de ocorrencias por janela de 30 min, guardadas como float64. Sao esparsas
  (maioria zero) e com assimetria muito alta.
- Slowness_in_traffic_percent: numerica continua, o alvo natural do dataset
  (percentual de lentidao no transito na janela de 30 min).
"""

print("\ndados ausentes e duplicados\n".upper())
missing_data_report(df)

print("\nmedidas de tendencia central e dispersao\n".upper())
numeric_columns = df.select_dtypes("number").columns
print(numeric_summary(df, numeric_columns).round(2))

print("\nfoco na variavel alvo: slowness_in_traffic_percent\n".upper())
target = df["Slowness_in_traffic_percent"]
q1, q3 = target.quantile(0.25), target.quantile(0.75)
print(f"Media:          {target.mean():.2f}%")
print(f"Mediana:        {target.median():.2f}%")
print(f"Moda:           {target.mode().iloc[0]:.2f}%")
print(f"Desvio padrao:  {target.std():.2f} pontos percentuais")
print(f"Variancia:      {target.var():.2f}")
print(f"Q1 / Q3:        {q1:.2f}% / {q3:.2f}%")
print(f"IQR:            {q3 - q1:.2f}")
print(f"Assimetria:     {target.skew():.2f}")

print("\nboxplots e deteccao de outliers (iqr)\n".upper())
incident_columns = [c for c in df.columns if c not in ("Hour", "Slowness_in_traffic_percent")]

fig, axes = plt.subplots(4, 4, figsize=(16, 14))
for ax, column in zip(axes.flat, incident_columns):
    seaborn.boxplot(y=df[column], ax=ax, color="steelblue")
    ax.set_title(column, fontsize=9)
    ax.set_ylabel("")
for ax in axes.flat[len(incident_columns):]:
    ax.axis("off")
fig.suptitle("Boxplots das colunas de incidentes de transito (contagens por janela de 30 min)")
save_figure(OUTPUT_DIR, "boxplots_incidentes.png")

plt.figure(figsize=(6, 5))
seaborn.boxplot(y=target, color="orange")
plt.title("Boxplot: % de Lentidao no Transito")
save_figure(OUTPUT_DIR, "boxplot_lentidao.png")

print(outlier_report(df, numeric_columns))
"""
A maioria das colunas de incidentes acumula outliers pelo criterio de IQR
porque sao eventos raros (quase todo o tempo zero) -- nesse caso o "outlier"
e o proprio sinal de interesse (o incidente aconteceu), nao um erro de coleta.
"""

print("\ncorrelacao entre variaveis (heatmap)\n".upper())
correlation = df[numeric_columns].corr()
plt.figure(figsize=(12, 10))
seaborn.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 7})
plt.title("Matriz de Correlacao - Comportamento do Transito Urbano (Sao Paulo)")
save_figure(OUTPUT_DIR, "heatmap_correlacao.png")

corr_with_target = correlation["Slowness_in_traffic_percent"].drop("Slowness_in_traffic_percent").sort_values()
plt.figure(figsize=(8, 7))
colors = ["crimson" if v < 0 else "seagreen" for v in corr_with_target]
plt.barh(corr_with_target.index, corr_with_target.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Correlacao de cada tipo de incidente com % de Lentidao no Transito")
plt.xlabel("Coeficiente de correlacao (Pearson)")
save_figure(OUTPUT_DIR, "ranking_correlacao_alvo.png")
print(corr_with_target.sort_values(ascending=False))

print("\ngraficos de dispersao (scatter / pairplot)\n".upper())
top3 = corr_with_target.abs().sort_values(ascending=False).head(3).index.tolist()
pairplot_columns = ["Slowness_in_traffic_percent"] + top3
print("Colunas selecionadas para o pairplot:", pairplot_columns)

seaborn.pairplot(df[pairplot_columns], diag_kind="kde", corner=True)
plt.suptitle("Pairplot: Lentidao x principais incidentes correlacionados", y=1.02)
save_figure(OUTPUT_DIR, "pairplot_top3.png")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, incident in zip(axes, top3):
    seaborn.regplot(data=df, x=incident, y="Slowness_in_traffic_percent", ax=ax,
                     scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    ax.set_title(f"{incident} vs. Lentidao")
save_figure(OUTPUT_DIR, "scatter_top3.png")

print("\nanalise temporal: horario x dia da semana\n".upper())
weekdays = ["Segunda", "Terca", "Quarta", "Quinta", "Sexta"]
df["Weekday"] = pandas.Categorical([weekdays[i // 27] for i in range(len(df))], categories=weekdays, ordered=True)
hour_order = sorted(df["Hour"].unique(), key=lambda h: tuple(map(int, h.split(":"))))

mean_by_hour = df.groupby("Hour", observed=True)["Slowness_in_traffic_percent"].mean().reindex(hour_order)
plt.figure(figsize=(12, 5))
mean_by_hour.plot(marker="o")
plt.title("Lentidao media no transito por horario (media dos 5 dias)")
plt.ylabel("% de Lentidao (media)")
plt.xlabel("Horario")
plt.xticks(rotation=45)
save_figure(OUTPUT_DIR, "linha_media_por_horario.png")

plt.figure(figsize=(9, 5))
seaborn.boxplot(data=df, x="Weekday", y="Slowness_in_traffic_percent", hue="Weekday", palette="crest", legend=False)
seaborn.stripplot(data=df, x="Weekday", y="Slowness_in_traffic_percent", color="black", alpha=0.4, size=3)
plt.title("Distribuicao da Lentidao no Transito por Dia da Semana")
save_figure(OUTPUT_DIR, "boxplot_por_dia_semana.png")

pivot = df.pivot_table(index="Hour", columns="Weekday", values="Slowness_in_traffic_percent", observed=True)
pivot = pivot.reindex(hour_order)
plt.figure(figsize=(9, 8))
seaborn.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "% Lentidao"})
plt.title("Lentidao no transito: Horario x Dia da Semana")
save_figure(OUTPUT_DIR, "heatmap_horario_dia.png")
"""
O grafico de linha mostra os dois picos de transito esperados (fim da manha e
fim da tarde), com um vale no horario de almoco. O heatmap horario x dia mostra
que quarta e quinta-feira concentram os piores picos noturnos (>20% as 19h-20h),
bem acima de segunda-feira -- um padrao que so aparece cruzando as duas
dimensoes categoricas/ordinais.
"""

print("\nimpacto de outliers e dados ausentes em analises multivariadas (pesquisa)\n".upper())
print("""
Outliers:
- Distorcem media e desvio padrao (nao sao medidas robustas); um unico valor
  extremo desloca a media e infla o desvio, fazendo a distribuicao parecer mais
  dispersa do que e para a maioria dos dados. Mediana e IQR sofrem bem menos.
- Podem criar ou mascarar correlacoes em analise multivariada: um ponto extremo
  pode "puxar" uma reta de regressao ou o coeficiente de Pearson, inflando uma
  correlacao artificial ou escondendo o padrao real do restante dos dados.
- Distorcem distancias em algoritmos como k-means, KNN e PCA, que dependem de
  distancia euclidiana -- um outlier pode criar um cluster "fantasma" ou puxar
  componentes principais na sua direcao.
- Outlier nem sempre e erro: em series de contagem de eventos raros (como as
  colunas de incidentes deste dataset), o "outlier" pelo criterio de IQR e
  justamente o evento mais relevante para o problema, nao ruido de coleta.

Dados ausentes:
- Se a ausencia nao for aleatoria (ex.: sensor falha justamente durante um
  evento extremo), descartar linhas (dropna) enviesa a amostra, subestimando
  riscos e correlacoes nos casos mais criticos.
- Reduzem o tamanho amostral e o poder estatistico disponivel para estimar
  correlacoes e ajustar modelos, especialmente em datasets pequenos.
- Quando colunas diferentes tem ausencias em linhas diferentes, o calculo de
  correlacao par-a-par (pairwise deletion, padrao do pandas .corr()) usa
  subconjuntos distintos de linhas por par de colunas, podendo gerar matrizes
  de correlacao inconsistentes -- o que quebra tecnicas como PCA.
- Imputar com media/mediana reduz artificialmente a variancia da coluna, o que
  tambem distorce desvio padrao e correlacoes; metodos mais sofisticados (KNN
  imputer, imputacao multipla) preservam melhor a estrutura multivariada, mas
  sempre assumem algo sobre o mecanismo de ausencia dos dados.

Neste dataframe (Comportamento do Transito Urbano de Sao Paulo) nao ha valores
ausentes nem duplicatas, entao esse problema especifico nao se aplica aqui --
mas o problema dos outliers "genuinos" nas colunas de incidentes e real e foi
o principal achado desta analise.
""")

print("\ngerando relatorio html\n".upper())
build_html_report(OUTPUT_DIR, title="Comportamento do Transito Urbano - Sao Paulo")
