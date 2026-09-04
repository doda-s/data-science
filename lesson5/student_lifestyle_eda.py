"""
Link to dataset https://www.openml.org/search?type=data&status=active&id=46873
Daily Lifestyle and Academic Performance of Students: 2000 estudantes, com horas
diarias dedicadas a estudo, atividades extracurriculares, sono, vida social e
atividade fisica, alem do GPA (nota media academica). A fonte ja remove
Student_ID por nao ser relevante; esta versao do OpenML tambem nao traz a
coluna Stress_Level mencionada na descricao original do dataset.

Analise exploratoria e multivariada deste dataframe: estatisticas descritivas
(media, mediana, moda, desvio padrao, variancia, quartis), deteccao de outliers
(boxplot + IQR), dados ausentes/duplicados, correlacao (heatmap), regressao
linear formal (scipy.stats.linregress), graficos de dispersao, violinplots por
faixa de GPA -- e uma verificacao estrutural rara: as 5 colunas de horas diarias
sempre somam exatamente 24, o que faz deste um dataset "composicional" (closed
data), com implicacoes diretas para a interpretacao de correlacoes. Os graficos
sao salvos em lesson5/output/student_lifestyle/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils.eda_helpers import setup_output_dir, save_figure, numeric_summary, outlier_report, missing_data_report
from utils.html_report import build_html_report

import matplotlib.pyplot as plt
import pandas
import scipy.stats as stats
import seaborn
import sklearn.datasets as skdatasets

OUTPUT_DIR = setup_output_dir(__file__, "student_lifestyle")

print("Fetching dataset...")
df: pandas.DataFrame = skdatasets.fetch_openml(data_id=46873, as_frame=True).data
print("Dataset fetched!")

print("\nestrutura e tipos\n".upper())
print(df.info())
print(df.head(10))

"""
Features:
- Study_Hours_Per_Day, Extracurricular_Hours_Per_Day, Sleep_Hours_Per_Day,
  Social_Hours_Per_Day, Physical_Activity_Hours_Per_Day: numericas continuas,
  cada uma o numero de horas por dia dedicado aquela atividade.
- GPA: numerica continua, o alvo (nota media academica, escala tipica 0-4).
Nao ha nenhuma coluna categorica neste dataframe.
"""

print("\ndados ausentes e duplicados\n".upper())
missing_data_report(df)

print("\nmedidas de tendencia central e dispersao\n".upper())
numeric_columns = df.columns.tolist()
print(numeric_summary(df, numeric_columns).round(2))

print("\nfoco na variavel alvo: gpa\n".upper())
target = df["GPA"]
q1, q3 = target.quantile(0.25), target.quantile(0.75)
print(f"Media:          {target.mean():.2f}")
print(f"Mediana:        {target.median():.2f}")
print(f"Moda:           {target.mode().iloc[0]:.2f} (contagem: {target.value_counts().iloc[0]})")
print(f"Desvio padrao:  {target.std():.2f}")
print(f"Variancia:      {target.var():.2f}")
print(f"Q1 / Q3:        {q1:.2f} / {q3:.2f}")
print(f"IQR:            {q3 - q1:.2f}")
print(f"Assimetria:     {target.skew():.2f}")

print("\nverificacao estrutural: dataset composicional (soma = 24h/dia)\n".upper())
hour_columns = [c for c in numeric_columns if c != "GPA"]
total_hours = df[hour_columns].sum(axis=1)
print(total_hours.describe())
n_diff_from_24 = (total_hours.sub(24).abs() > 1e-6).sum()
print(f"Linhas onde a soma das 5 colunas de horas difere de 24: {n_diff_from_24} / {len(df)}")
"""
As 5 colunas de horas SEMPRE somam exatamente 24 -- ou seja, nao sao 5 medidas
independentes, e sim uma composicao fechada do dia de cada estudante (dados
"composicionais"). Isso tem uma consequencia matematica direta: qualquer uma
das 5 colunas pode ser escrita como "24 menos a soma das outras 4", entao as 5
juntas sao linearmente dependentes (a matriz de dados tem posto no maximo 4,
nao 5). Se todas as 5 fossem usadas ao mesmo tempo como preditoras em uma
regressao linear multipla, o sistema seria singular (multicolinearidade
perfeita) -- seria preciso remover uma coluna ou usar tecnicas especificas para
dados composicionais (ver secao de pesquisa ao final).
"""

print("\nboxplots e deteccao de outliers (iqr)\n".upper())
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, column in zip(axes.flat, numeric_columns):
    seaborn.boxplot(y=df[column], ax=ax, color="steelblue")
    ax.set_title(column, fontsize=9)
    ax.set_ylabel("")
for ax in axes.flat[len(numeric_columns):]:
    ax.axis("off")
fig.suptitle("Boxplots das variaveis de estilo de vida e GPA")
save_figure(OUTPUT_DIR, "boxplots_variaveis.png")

print(outlier_report(df, numeric_columns))
"""
Praticamente nao ha outliers (no maximo 0.25% das linhas em qualquer coluna).
Faz sentido: como as 5 horas diarias sao limitadas pelo teto fixo de 24h/dia,
nenhuma delas pode assumir valores extremos isolados sem "roubar" horas de
outra atividade -- a propria restricao estrutural do dataset already limita a
amplitude das variaveis, ao contrario dos datasets anteriores.
"""

print("\ncorrelacao entre variaveis (heatmap)\n".upper())
correlation = df[numeric_columns].corr()
plt.figure(figsize=(8, 6))
seaborn.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlacao - Estilo de Vida e GPA")
save_figure(OUTPUT_DIR, "heatmap_correlacao.png")

corr_with_target = correlation["GPA"].drop("GPA").sort_values()
plt.figure(figsize=(8, 5))
colors = ["crimson" if v < 0 else "seagreen" for v in corr_with_target]
plt.barh(corr_with_target.index, corr_with_target.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Correlacao de cada variavel de estilo de vida com o GPA")
plt.xlabel("Coeficiente de correlacao (Pearson)")
save_figure(OUTPUT_DIR, "ranking_correlacao_alvo.png")
print(corr_with_target.sort_values(ascending=False))
"""
Study_Hours_Per_Day tem a correlacao mais forte e positiva com o GPA (0.73),
enquanto Physical_Activity_Hours_Per_Day e a mais negativa (-0.34). Mas como
vimos acima, as 5 colunas somam sempre 24 -- entao parte dessas correlacoes,
principalmente as negativas, pode ser um artefato da "restricao de fechamento"
(closure): se um estudante estuda mais horas, matematicamente sobra menos
tempo para as outras atividades, o que por si so ja geraria correlacoes
negativas entre elas mesmo que o comportamento real de cada aluno fosse
independente. Isso NAO significa que a correlacao com Study_Hours seja falsa
(o efeito e forte e consistente com a intuicao), mas reforca que, em dados
composicionais, correlacoes de Pearson comuns devem ser interpretadas com
cautela (mais detalhes na secao de pesquisa).
"""

print("\nregressao linear formal: study_hours_per_day -> gpa\n".upper())
regression = stats.linregress(df["Study_Hours_Per_Day"], df["GPA"])
print(f"Coeficiente angular (slope): {regression.slope:.4f} pontos de GPA por hora extra de estudo")
print(f"Intercepto:                  {regression.intercept:.4f}")
print(f"R (correlacao):               {regression.rvalue:.4f}")
print(f"R^2 (variancia explicada):   {regression.rvalue ** 2:.4f}")
print(f"p-valor:                      {regression.pvalue:.2e}")
print(f"Erro padrao do slope:         {regression.stderr:.4f}")
"""
scipy.stats.linregress complementa o regplot do seaborn com numeros formais:
o slope (~0.15) diz que, em media, cada hora extra de estudo por dia esta
associada a um aumento de ~0.15 ponto no GPA; R^2 (~0.54) diz que Study_Hours
sozinha explica cerca de 54% da variancia do GPA; e o p-valor extremamente
baixo confirma que essa relacao nao e fruto do acaso nesta amostra de 2000
estudantes.
"""

print("\ngraficos de dispersao (scatter / pairplot)\n".upper())
top3 = corr_with_target.abs().sort_values(ascending=False).head(3).index.tolist()
pairplot_columns = ["GPA"] + top3
print("Colunas selecionadas para o pairplot:", pairplot_columns)

seaborn.pairplot(df[pairplot_columns], diag_kind="kde", corner=True)
plt.suptitle("Pairplot: GPA x principais variaveis correlacionadas", y=1.02)
save_figure(OUTPUT_DIR, "pairplot_top3.png")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, column in zip(axes, top3):
    seaborn.regplot(data=df, x=column, y="GPA", ax=ax,
                     scatter_kws={"alpha": 0.4, "s": 12}, line_kws={"color": "red"})
    ax.set_title(f"{column} vs. GPA")
save_figure(OUTPUT_DIR, "scatter_top3.png")

print("\nanalise por faixa de desempenho: violinplots por tercil de gpa\n".upper())
df["GPA_Tier"] = pandas.qcut(df["GPA"], q=3, labels=["Baixo", "Medio", "Alto"])
print(df["GPA_Tier"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
seaborn.violinplot(data=df, x="GPA_Tier", y="Study_Hours_Per_Day", hue="GPA_Tier",
                    order=["Baixo", "Medio", "Alto"], palette="crest", legend=False, ax=axes[0])
axes[0].set_title("Horas de Estudo por Faixa de GPA")

seaborn.violinplot(data=df, x="GPA_Tier", y="Physical_Activity_Hours_Per_Day", hue="GPA_Tier",
                    order=["Baixo", "Medio", "Alto"], palette="mako", legend=False, ax=axes[1])
axes[1].set_title("Atividade Fisica por Faixa de GPA")
save_figure(OUTPUT_DIR, "violinplot_faixas_gpa.png")
"""
Os violinplots mostram nao so a diferenca de mediana entre as faixas de GPA
(como um boxplot mostraria), mas tambem o formato completo da distribuicao em
cada faixa: os estudantes da faixa "Alto" GPA concentram sua distribuicao de
Study_Hours em valores mais altos e mais estreitos, enquanto a faixa "Baixo"
tem uma distribuicao mais espalhada e deslocada para menos horas de estudo --
o inverso acontece com Atividade Fisica, reforcando a correlacao negativa.
"""

print("\nresumo das features\n".upper())
print("""
Grupo de estilo de vida (horas/dia, sempre somando 24): Study_Hours_Per_Day,
    Extracurricular_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day,
    Physical_Activity_Hours_Per_Day -- todas numericas continuas, dado
    composicional (closed data).
Alvo: GPA (numerica continua, 2.24 a 4.0), correlacionada principalmente com
    Study_Hours_Per_Day (+0.73) e Physical_Activity_Hours_Per_Day (-0.34).
Nao ha colunas categoricas, nulos, duplicatas ou outliers relevantes.
""")

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
- Neste dataset praticamente nao ha outliers, em parte porque a propria
  estrutura dos dados (horas limitadas a um teto de 24h/dia) ja restringe a
  amplitude possivel de cada variavel.

Dados ausentes:
- Se a ausencia nao for aleatoria, descartar linhas (dropna) enviesa a amostra.
- Reduzem o tamanho amostral e o poder estatistico disponivel para estimar
  correlacoes e ajustar modelos.
- Quando colunas diferentes tem ausencias em linhas diferentes, o calculo de
  correlacao par-a-par (pairwise deletion, padrao do pandas .corr()) usa
  subconjuntos distintos de linhas por par de colunas, podendo gerar matrizes
  de correlacao inconsistentes -- o que quebra tecnicas como PCA.
- Imputar com media/mediana reduz artificialmente a variancia da coluna, o que
  tambem distorce desvio padrao e correlacoes.

Dados composicionais (o achado central deste dataset):
- Quando um grupo de variaveis sempre soma uma constante (aqui, 24 horas), elas
  carregam "informacao compartilhada" por construcao: aumentar uma obriga a
  diminuir outra(s). Isso e conhecido na literatura estatistica como o
  "problema do fechamento" (closure problem), descrito originalmente por Karl
  Pearson (1897) como "correlacao espuria" e formalizado depois por John
  Aitchison (1986) na area de Compositional Data Analysis (CoDA).
- Consequencia pratica: correlacoes de Pearson calculadas diretamente sobre
  dados composicionais tendem a ser artificialmente mais negativas do que a
  relacao real entre as variaveis, e usar todas as componentes como preditoras
  em uma regressao linear multipla cria multicolinearidade perfeita (posto
  deficiente na matriz de dados).
- A forma correta de analisar dados assim costuma envolver transformacoes de
  log-razao (ex.: centered log-ratio, isometric log-ratio) antes de aplicar
  correlacao, regressao ou PCA -- fora do escopo deste script, mas essencial
  de se conhecer antes de tirar conclusoes causais de datasets de "orcamento
  de tempo" como este (horas do dia, alocacao de orcamento, composicao de
  nutrientes em uma dieta, etc. sao exemplos classicos de dados
  composicionais).
""")

print("\ngerando relatorio html\n".upper())
build_html_report(OUTPUT_DIR, title="Student Lifestyle & Academic Performance - EDA")
