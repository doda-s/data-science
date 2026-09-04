"""
Link to dataset https://www.openml.org/search?type=data&status=active&id=43098
Students Performance in Exams: 1000 estudantes, com 4 atributos categoricos de
contexto socioeconomico/academico (etnia anonimizada, escolaridade dos pais,
tipo de almoco -- proxy de nivel socioeconomico nos EUA -- e conclusao ou nao
de um curso preparatorio) e 3 notas continuas (matematica, leitura, escrita,
0-100). Esta versao do OpenML nao inclui a coluna "gender" presente na versao
original do Kaggle -- outra diferenca de esquema entre fontes do mesmo dataset.

Analise exploratoria e multivariada deste dataframe: renomeacao de colunas
(nomes com pontos nao sao idiomaticos em pandas/Python), estatisticas
descritivas (media, mediana, moda, desvio padrao, variancia, quartis),
deteccao de outliers (boxplot + IQR), dados ausentes/duplicados, correlacao
(heatmap), engenharia de atributo (nota media + aprovado/reprovado), graficos
de dispersao com hue categorico, interacao entre 2 variaveis categoricas e
1 numerica, e uma nota sobre cuidados eticos ao interpretar atributos
demograficos. Os graficos sao salvos em lesson5/output/students_score/.
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

OUTPUT_DIR = setup_output_dir(__file__, "students_score")

print("Fetching dataset...")
df: pandas.DataFrame = skdatasets.fetch_openml(data_id=43098, as_frame=True).data
print("Dataset fetched!")

print("\nestrutura e tipos (antes da limpeza)\n".upper())
print(df.info())
print(df.head(10))

print("\nlimpeza de esquema: nomes de colunas com pontos\n".upper())
df = df.rename(columns={
    "race.ethnicity": "Race_Ethnicity",
    "parental.level.of.education": "Parental_Education",
    "lunch": "Lunch",
    "test.preparation.course": "Test_Prep_Course",
    "math.score": "Math_Score",
    "reading.score": "Reading_Score",
    "writing.score": "Writing_Score",
})
print("Colunas renomeadas para Title_Case sem pontos (ex.: 'math.score' -> 'Math_Score').")
print(df.columns.tolist())

"""
Features:
- Race_Ethnicity: categorica nominal, 5 grupos anonimizados ("group A".."group E").
- Parental_Education: categorica ORDINAL, 6 niveis (some high school < high
  school < some college < associate's degree < bachelor's degree < master's degree).
- Lunch: categorica binaria (standard / free-reduced) -- nos EUA, elegibilidade
  para almoco gratuito/reduzido e um proxy classico de nivel socioeconomico
  familiar usado em pesquisa educacional.
- Test_Prep_Course: categorica binaria (completed / none).
- Math_Score, Reading_Score, Writing_Score: numericas continuas (0-100).
"""

print("\ndados ausentes e duplicados\n".upper())
missing_data_report(df)

print("\nmedidas de tendencia central e dispersao\n".upper())
score_columns = ["Math_Score", "Reading_Score", "Writing_Score"]
print(numeric_summary(df, score_columns).round(2))

print("\nfoco na variavel math_score\n".upper())
target = df["Math_Score"]
q1, q3 = target.quantile(0.25), target.quantile(0.75)
print(f"Media:          {target.mean():.2f}")
print(f"Mediana:        {target.median():.2f}")
print(f"Moda:           {target.mode().iloc[0]:.2f} (contagem: {target.value_counts().iloc[0]})")
print(f"Desvio padrao:  {target.std():.2f}")
print(f"Variancia:      {target.var():.2f}")
print(f"Q1 / Q3:        {q1:.2f} / {q3:.2f}")
print(f"IQR:            {q3 - q1:.2f}")
print(f"Assimetria:     {target.skew():.2f} (leve assimetria negativa: cauda mais longa para notas baixas)")

worst_student = df.loc[df["Math_Score"].idxmin()]
print("\nEstudante com a pior nota de matematica (0):")
print(worst_student)
"""
Nao e um valor isolado/erro de digitacao: o mesmo estudante tambem tem as
piores notas de leitura (17) e escrita (10) do dataset inteiro -- ou seja, e
um caso real de desempenho consistentemente baixo nas 3 provas, nao um outlier
espurio de uma unica materia.
"""

print("\nboxplots e deteccao de outliers (iqr)\n".upper())
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, column in zip(axes, score_columns):
    seaborn.boxplot(y=df[column], ax=ax, color="steelblue")
    ax.set_title(column, fontsize=10)
    ax.set_ylabel("")
fig.suptitle("Boxplots das notas (Math / Reading / Writing)")
save_figure(OUTPUT_DIR, "boxplots_notas.png")

print(outlier_report(df, score_columns))
"""
Os poucos outliers encontrados (5 a 8 por materia, <1% do dataset) estao todos
na cauda inferior -- alunos com desempenho muito abaixo da media, coerente com
a assimetria negativa das 3 distribuicoes. Nenhum outlier esta no teto (nota
100), o que faz sentido: 100 e o valor maximo possivel, entao nao ha "excesso"
acima dele por definicao.
"""

print("\ncorrelacao entre as notas (heatmap)\n".upper())
correlation = df[score_columns].corr()
plt.figure(figsize=(6, 5))
seaborn.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".3f")
plt.title("Matriz de Correlacao - Notas dos Estudantes")
save_figure(OUTPUT_DIR, "heatmap_correlacao.png")
print(correlation.round(3))
"""
Reading_Score e Writing_Score sao quase perfeitamente correlacionadas (0.955)
-- as duas provas medem habilidades muito proximas (compreensao e producao de
texto), quase redundantes estatisticamente. Math_Score correlaciona um pouco
menos com as outras duas (~0.80-0.82), sugerindo que raciocinio matematico
compartilha alguma habilidade geral de "desempenho academico" com leitura e
escrita, mas tambem tem uma componente propria.
"""

print("\nengenharia de atributos: nota media e aprovado/reprovado\n".upper())
df["Average_Score"] = df[score_columns].mean(axis=1)
df["Passed"] = df["Average_Score"] >= 60
print(df[["Average_Score", "Passed"]].describe(include="all"))
print(f"Taxa de aprovacao (media >= 60): {df['Passed'].mean():.1%}")
"""
Average_Score resume as 3 provas em um unico indicador de desempenho, e Passed
converte esse indicador continuo em uma decisao binaria usando um corte fixo
do mundo real (60 pontos), diferente da abordagem por quantis (tercis) usada
no dataset de estilo de vida -- aqui faz mais sentido um corte fixo porque
existe um criterio de aprovacao conhecido, e nao apenas uma comparacao relativa
entre estudantes da propria amostra.
"""

print("\ngraficos de dispersao com hue categorico (pairplot / scatter)\n".upper())
seaborn.pairplot(df[score_columns + ["Test_Prep_Course"]], hue="Test_Prep_Course", diag_kind="kde", corner=True)
plt.suptitle("Pairplot das notas, colorido por curso preparatorio", y=1.02)
save_figure(OUTPUT_DIR, "pairplot_notas_por_prep.png")

plt.figure(figsize=(7, 6))
seaborn.scatterplot(data=df, x="Reading_Score", y="Writing_Score", hue="Lunch", alpha=0.6)
plt.title("Reading Score vs. Writing Score, colorido por tipo de almoco")
save_figure(OUTPUT_DIR, "scatter_reading_writing_por_lunch.png")

print("\nanalise categorica: preparo, almoco, escolaridade dos pais e etnia\n".upper())
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
seaborn.violinplot(data=df, x="Test_Prep_Course", y="Average_Score", hue="Test_Prep_Course", palette="crest", legend=False, ax=axes[0])
axes[0].set_title("Nota Media por Curso Preparatorio")
seaborn.violinplot(data=df, x="Lunch", y="Average_Score", hue="Lunch", palette="mako", legend=False, ax=axes[1])
axes[1].set_title("Nota Media por Tipo de Almoco")
save_figure(OUTPUT_DIR, "violinplot_prep_lunch.png")

education_order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
df["Parental_Education"] = pandas.Categorical(df["Parental_Education"], categories=education_order, ordered=True)
mean_by_education = df.groupby("Parental_Education", observed=True)["Average_Score"].mean().reindex(education_order)
plt.figure(figsize=(9, 5))
mean_by_education.plot(marker="o")
plt.title("Nota Media por Nivel de Escolaridade dos Pais (ordenado)")
plt.ylabel("Average_Score (media)")
plt.xticks(rotation=20)
save_figure(OUTPUT_DIR, "linha_media_por_escolaridade_pais.png")

plt.figure(figsize=(9, 5))
seaborn.boxplot(data=df, x="Race_Ethnicity", y="Average_Score", hue="Race_Ethnicity",
                 order=sorted(df["Race_Ethnicity"].unique()), palette="viridis", legend=False)
plt.title("Nota Media por Grupo de Etnia (rotulos anonimizados pela fonte)")
save_figure(OUTPUT_DIR, "boxplot_media_por_etnia.png")
"""
Interpretacao com cautela: "Race_Ethnicity" usa rotulos anonimizados
("group A".."group E") sem contexto adicional fornecido pela fonte sobre
metodologia de coleta ou definicao dos grupos. Diferencas de media entre
grupos podem refletir fatores socioeconomicos e estruturais correlacionados
com etnia (como o proprio "Lunch" e "Parental_Education" ja sugerem), nao uma
relacao causal com a etnia em si. Datasets demograficos como este exigem
cuidado etico redobrado: correlacao entre um atributo demografico e uma
metrica de desempenho nao deve ser usada para conclusoes causais ou
generalizacoes sem o contexto socioeconomico completo por tras dos dados.
"""

print("\nanalise de interacao: curso preparatorio x tipo de almoco\n".upper())
interaction = df.groupby(["Lunch", "Test_Prep_Course"], observed=True)["Average_Score"].mean().unstack()
print(interaction.round(2))

seaborn.catplot(data=df, x="Lunch", y="Average_Score", hue="Test_Prep_Course",
                 kind="bar", palette="Set2", height=5, aspect=1.4)
plt.title("Interacao: Nota Media por Almoco x Curso Preparatorio")
save_figure(OUTPUT_DIR, "catplot_interacao_lunch_prep.png")
"""
As duas variaveis categoricas parecem ter efeitos aditivos (nao interativos)
sobre a nota media: tanto ter almoco padrao quanto ter completado o curso
preparatorio aumentam a media, e o grupo com AMBOS ("standard" + "completed")
tem a maior media combinada, sem um efeito de interacao surpreendente (isto e,
o "bonus" do curso preparatorio e parecido nos dois grupos de almoco).
"""

print("\nresumo das features\n".upper())
print("""
Grupo categorico:  Race_Ethnicity (nominal, 5 grupos anonimizados),
                   Parental_Education (ordinal, 6 niveis),
                   Lunch (binaria, proxy socioeconomico),
                   Test_Prep_Course (binaria)
Grupo numerico:    Math_Score, Reading_Score, Writing_Score (0-100,
                   assimetria negativa leve, poucos outliers na cauda inferior)
Derivadas:         Average_Score (media das 3 notas), Passed (Average_Score >= 60)
Sem nulos, duplicatas ou coluna 'gender' (ausente nesta versao do dataset).
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
- Neste dataset os poucos outliers (notas muito baixas) sao desempenhos reais
  e legitimos de alunos, nao erros de digitacao -- removê-los enviesaria
  qualquer analise de equidade educacional ao esconder justamente os casos que
  mais precisam de atencao.

Dados ausentes:
- Se a ausencia nao for aleatoria (ex.: alunos que faltaram a prova tendem a
  ter registros incompletos), descartar linhas (dropna) enviesa a amostra.
- Reduzem o tamanho amostral e o poder estatistico disponivel para estimar
  correlacoes e ajustar modelos.
- Quando colunas diferentes tem ausencias em linhas diferentes, o calculo de
  correlacao par-a-par (pairwise deletion, padrao do pandas .corr()) usa
  subconjuntos distintos de linhas por par de colunas, podendo gerar matrizes
  de correlacao inconsistentes -- o que quebra tecnicas como PCA.
- Imputar com media/mediana reduz artificialmente a variancia da coluna, o que
  tambem distorce desvio padrao e correlacoes.

Um terceiro cuidado, especifico de datasets com atributos demograficos como
este: correlacao NAO implica causalidade, e atributos como Race_Ethnicity
frequentemente carregam correlacoes espurias herdadas de outras variaveis
socioeconomicas (aqui, Lunch e Parental_Education). Uma analise multivariada
responsavel deve controlar por essas variaveis de confusao antes de atribuir
qualquer diferenca de desempenho a um grupo demografico especifico -- ausencia
desse cuidado e uma das formas mais comuns de vies (bias) introduzido durante
a propria analise exploratoria, mesmo quando os dados em si estao completos e
sem outliers.
""")

print("\ngerando relatorio html\n".upper())
build_html_report(OUTPUT_DIR, title="Students Performance in Exams - EDA")
