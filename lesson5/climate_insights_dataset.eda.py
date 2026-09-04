"""
Link to dataset https://www.openml.org/search?type=data&status=active&id=46729
Climate Insights Dataset: 10000 registros sinteticos (2000-2022) de Temperatura,
Emissao de CO2, Precipitacao, Umidade e Velocidade do Vento, por localidade/pais.

Analise exploratoria e multivariada deste dataframe: limpeza de texto (valores
vindos com aspas literais), parsing de datas, estatisticas descritivas (media,
mediana, moda, desvio padrao, variancia, quartis), deteccao de outliers
(boxplot + IQR), dados ausentes/duplicados, correlacao (heatmap), graficos de
dispersao, analise temporal (por ano) e analise geografica (por pais). Os
graficos sao salvos em lesson5/output/climate_insights/.
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

OUTPUT_DIR = setup_output_dir(__file__, "climate_insights")

print("Fetching dataset...")
df: pandas.DataFrame = skdatasets.fetch_openml(data_id=46729, as_frame=True).data
print("Dataset fetched!")

print("\nestrutura e tipos (antes da limpeza)\n".upper())
print(df.info())
print(df.head(10))

"""
Features:
- Date: veio como string, sempre entre aspas literais ("'2000-01-01 ...'").
  Apos limpeza vira datetime; dela derivamos Year e Month.
- Location: string, quase um identificador (7764 valores unicos em 10000
  linhas) -- pouco util para agrupamento, mas mostra cardinalidade alta.
- Country: string categorica com 243 categorias.
- Temperature, CO2 Emissions, Precipitation, Humidity, Wind Speed: numericas
  continuas (float64).
"""

print("\nlimpeza de texto: aspas literais nos valores\n".upper())
quoted_before = {
    "Date": df["Date"].str.startswith("'").sum(),
    "Location": df["Location"].str.startswith("'").sum(),
    "Country": df["Country"].str.startswith("'").sum(),
}
print(f"Linhas com aspas literais por coluna (antes da limpeza): {quoted_before}")

for column in ("Date", "Location", "Country"):
    df[column] = df[column].str.strip("'")

df["Date"] = pandas.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
print("Aspas removidas de Date/Location/Country; Date convertida para datetime.")
print(f"Novas colunas derivadas: Year (min={df['Year'].min()}, max={df['Year'].max()}), Month.")
print(df[["Date", "Location", "Country", "Year", "Month"]].head(5))
"""
Isso NAO criou categorias duplicadas neste dataset especifico (cada pais
aparece sempre com ou sempre sem aspas), mas e um risco real: se o mesmo pais
aparecesse ora como "Chad" ora como "'Chad'", um groupby ou value_counts trataria
como duas categorias diferentes, inflando artificialmente a cardinalidade e
distorcendo qualquer analise categorica/multivariada -- por isso a limpeza de
texto e feita antes de qualquer outra estatistica.
"""

print("\ndados ausentes e duplicados\n".upper())
missing_data_report(df)

print("\nmedidas de tendencia central e dispersao\n".upper())
numeric_columns = ["Temperature", "CO2 Emissions", "Precipitation", "Humidity", "Wind Speed"]
print(numeric_summary(df, numeric_columns).round(2))

n_unique = {c: df[c].nunique() for c in numeric_columns}
print(f"Valores unicos por coluna: {n_unique}")
"""
As 5 colunas tem 10000 valores unicos em 10000 linhas -- ou seja, nenhum valor
se repete. Nesse cenario a moda NAO e uma medida informativa: o pandas apenas
retorna o menor valor entre os empatados em frequencia 1 (na pratica, o minimo
da coluna). Para variaveis continuas como estas, media e mediana sao as medidas
de tendencia central que realmente importam; a moda so e util quando ha valores
genuinamente repetidos (como nas contagens de incidentes do dataset da lesson 1).
"""

print("\nfoco em temperature e co2 emissions\n".upper())
for column in ("Temperature", "CO2 Emissions"):
    series = df[column]
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    print(f"--- {column} ---")
    print(f"Media:          {series.mean():.2f}")
    print(f"Mediana:        {series.median():.2f}")
    print(f"Desvio padrao:  {series.std():.2f}")
    print(f"Variancia:      {series.var():.2f}")
    print(f"Q1 / Q3:        {q1:.2f} / {q3:.2f}")
    print(f"IQR:            {q3 - q1:.2f}")
    print(f"Assimetria:     {series.skew():.2f}")

print("\nboxplots e deteccao de outliers (iqr)\n".upper())
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, column in zip(axes, numeric_columns):
    seaborn.boxplot(y=df[column], ax=ax, color="steelblue")
    ax.set_title(column, fontsize=10)
    ax.set_ylabel("")
fig.suptitle("Boxplots das variaveis climaticas (Climate Insights Dataset)")
save_figure(OUTPUT_DIR, "boxplots_variaveis_climaticas.png")

print(outlier_report(df, numeric_columns))
"""
Temperature e CO2 Emissions tem uma fracao pequena (~0.8%) de outliers pelo
criterio de IQR -- exatamente o esperado para uma distribuicao normal bem
comportada, que sempre deixa uma cauda fina alem de 1.5*IQR. Precipitation,
Humidity e Wind Speed nao tem nenhum outlier, porque sao proximas de uniformes
entre 0 e 100 (sem cauda para gerar valores extremos). Diferente do dataset de
transito urbano (lesson 1), aqui nao ha outliers "genuinos" fora do esperado
estatisticamente -- um bom contraste: nem todo dataset tem outliers relevantes,
e a ausencia deles tambem e uma informacao sobre a natureza (sintetica e bem
comportada) dos dados.
"""

print("\ncorrelacao entre variaveis (heatmap)\n".upper())
correlation = df[numeric_columns].corr()
plt.figure(figsize=(8, 6))
seaborn.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlacao - Climate Insights Dataset")
save_figure(OUTPUT_DIR, "heatmap_correlacao.png")
print(correlation.round(3))
"""
Todas as correlacoes ficam muito proximas de zero (|r| < 0.03). Isso reforca
que o dataset e sintetico e gerado de forma independente entre as variaveis --
em dados climaticos reais esperariamos, por exemplo, alguma correlacao entre
CO2 Emissions e Temperature ao longo do tempo. A ausencia de correlacao aqui e
o proprio achado: nem toda analise multivariada revela relacoes fortes, e
reconhecer "ruido" e tao importante quanto reconhecer "sinal".
"""

print("\ngraficos de dispersao (scatter / pairplot)\n".upper())
seaborn.pairplot(df[numeric_columns], diag_kind="kde", corner=True)
plt.suptitle("Pairplot: variaveis climaticas", y=1.02)
save_figure(OUTPUT_DIR, "pairplot_variaveis_climaticas.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
seaborn.regplot(data=df, x="CO2 Emissions", y="Temperature", ax=axes[0],
                 scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "red"})
axes[0].set_title("CO2 Emissions vs. Temperature")
seaborn.regplot(data=df, x="Humidity", y="Wind Speed", ax=axes[1],
                 scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "red"})
axes[1].set_title("Humidity vs. Wind Speed")
save_figure(OUTPUT_DIR, "scatter_pares_selecionados.png")

print("\nanalise temporal: tendencia anual\n".upper())
yearly_mean = df.groupby("Year")[["Temperature", "CO2 Emissions"]].mean()
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.plot(yearly_mean.index, yearly_mean["Temperature"], marker="o", color="tab:red", label="Temperature")
ax2.plot(yearly_mean.index, yearly_mean["CO2 Emissions"], marker="s", color="tab:blue", label="CO2 Emissions")
ax1.set_xlabel("Ano")
ax1.set_ylabel("Temperature (media)", color="tab:red")
ax2.set_ylabel("CO2 Emissions (media)", color="tab:blue")
plt.title("Media anual de Temperatura e Emissao de CO2 (2000-2022)")
save_figure(OUTPUT_DIR, "tendencia_anual_temperatura_co2.png")
print(yearly_mean.round(2))
"""
Assim como no heatmap, as medias anuais oscilam em uma faixa estreita sem
tendencia de subida ou queda visivel -- coerente com dados gerados de forma
independente do ano, ao contrario do que aconteceria com series historicas
reais de temperatura/CO2.
"""

print("\nanalise geografica: paises com mais registros\n".upper())
top_countries = df["Country"].value_counts().head(15)
plt.figure(figsize=(10, 6))
seaborn.barplot(x=top_countries.values, y=top_countries.index, hue=top_countries.index,
                 palette="viridis", legend=False)
plt.title("Top 15 paises por numero de registros")
plt.xlabel("Numero de registros")
save_figure(OUTPUT_DIR, "top_paises_registros.png")

top8_countries = top_countries.head(8).index
plt.figure(figsize=(11, 6))
seaborn.boxplot(data=df[df["Country"].isin(top8_countries)], x="Country", y="Temperature",
                 hue="Country", palette="crest", legend=False)
plt.title("Distribuicao de Temperatura nos 8 paises com mais registros")
plt.xticks(rotation=30)
save_figure(OUTPUT_DIR, "boxplot_temperatura_top8_paises.png")
"""
Location tem 7764 valores unicos em 10000 linhas -- quase um identificador de
linha, entao agrupar por Location nao agregaria informacao util. Country (243
categorias) e o nivel de agregacao geografica que faz sentido para comparar
distribuicoes entre grupos.
"""

print("\nresumo das features\n".upper())
print("""
Grupo temporal:     Date (datetime, derivada de string com aspas) -> Year, Month
Grupo geografico:   Country (categorica, 243 niveis), Location (quase unico, 7764/10000)
Grupo numerico:     Temperature, CO2 Emissions, Precipitation, Humidity, Wind Speed
                    (todas continuas, sem outliers relevantes, sem correlacao entre si)
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
- Neste dataset praticamente nao ha outliers (variaveis bem comportadas), o que
  tambem e informativo: confirma que os dados sao sinteticos e gerados por
  distribuicoes conhecidas (normal/uniforme), sem eventos extremos genuinos.

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

Alem de outliers e ausencia, este dataset trouxe um terceiro problema tao comum
quanto os dois primeiros: inconsistencia de formatacao de texto (aspas literais
dentro dos proprios valores de Date/Location/Country). Sem a limpeza feita na
secao 2, o parsing de datas teria falhado e, em datasets onde a mesma categoria
aparece ora com ora sem aspas, o groupby/value_counts contaria como categorias
diferentes -- um tipo de "dado sujo" que se comporta como dado ausente/incorreto
na pratica, mesmo sem ser tecnicamente um NaN.
""")

print("\ngerando relatorio html\n".upper())
build_html_report(OUTPUT_DIR, title="Climate Insights Dataset - EDA")
