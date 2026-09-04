"""
Link to dataset https://www.openml.org/search?type=data&status=active&id=43540
Fuel Consumption & CO2 Emissions by Vehicles: 679 modelos de carros do ano-modelo
2001, com especificacoes de motor e consumo de combustivel, e a emissao de CO2
(g/km) como alvo declarado pela propria fonte do dataset.

Analise exploratoria e multivariada deste dataframe: limpeza de esquema (colunas
duplicadas/sem nome vindas da fonte original, valores de texto com aspas
literais), estatisticas descritivas (media, mediana, moda, desvio padrao,
variancia, quartis), deteccao de outliers (boxplot + IQR), dados
ausentes/duplicados, correlacao (heatmap) e multicolinearidade, graficos de
dispersao e analise categorica (tipo de combustivel, transmissao, fabricante).
Os graficos sao salvos em lesson5/output/emissions_by_cars/.
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

OUTPUT_DIR = setup_output_dir(__file__, "emissions_by_cars")

print("Fetching dataset...")
df: pandas.DataFrame = skdatasets.fetch_openml(data_id=43540, as_frame=True).data
print("Dataset fetched!")

print("\nestrutura e tipos (antes da limpeza)\n".upper())
print(df.info())
print(df.head(10))

print("\nlimpeza de esquema: colunas duplicadas/sem nome e texto com aspas\n".upper())
"""
A fonte original tem duas colunas chamadas "MODEL" (ano do modelo e nome do
modelo); o pandas renomeou a segunda para "MODEL.1" para evitar colisao. Alem
disso, 3 colunas de consumo de combustivel vieram sem cabecalho ("Unnamed:_9",
"Unnamed:_10", "Unnamed:_11"). Renomeamos tudo para nomes legiveis, na ordem
padrao deste tipo de dataset (consumo na cidade, rodovia, combinado em L/100km
e combinado em mpg).
"""
df = df.rename(columns={
    "MODEL": "Model_Year",
    "MODEL.1": "Model",
    "MAKE": "Make",
    "VEHICLE_CLASS": "Vehicle_Class",
    "ENGINE_SIZE": "Engine_Size_L",
    "CYLINDERS": "Cylinders",
    "TRANSMISSION": "Transmission",
    "FUEL": "Fuel_Type",
    "FUEL_CONSUMPTION*": "Fuel_Consumption_City_L100km",
    "Unnamed:_9": "Fuel_Consumption_Hwy_L100km",
    "Unnamed:_10": "Fuel_Consumption_Comb_L100km",
    "Unnamed:_11": "Fuel_Consumption_Comb_mpg",
    "CO2_EMISSIONS": "CO2_Emissions_gkm",
})

for column in ("Model", "Vehicle_Class"):
    df[column] = df[column].str.strip("'")

fuel_labels = {"X": "Regular gasoline", "Z": "Premium gasoline", "D": "Diesel", "E": "Ethanol (E85)", "N": "Natural gas"}
df["Fuel_Type"] = df["Fuel_Type"].map(fuel_labels)

transmission_labels = {"A": "Automatic", "AS": "Automatic (Select Shift)", "AV": "CVT", "M": "Manual"}
df["Transmission_Type"] = df["Transmission"].str.extract(r"^([A-Z]+)")[0].map(transmission_labels)

print("Colunas renomeadas, aspas removidas de Model/Vehicle_Class, Fuel_Type e Transmission_Type decodificados.")
print(df[["Model", "Vehicle_Class", "Fuel_Type", "Transmission", "Transmission_Type"]].head(5))

"""
Features (apos a limpeza):
- Model_Year: numerica, mas constante (todas as 679 linhas sao do ano 2001) --
  esta fatia especifica do dataset nao permite analise temporal.
- Make (34 categorias), Model (351 categorias, quase um identificador),
  Vehicle_Class (14 categorias), Fuel_Type (5 categorias, decodificadas),
  Transmission / Transmission_Type (8 codigos agrupados em 4 categorias):
  todas categoricas nominais.
- Engine_Size_L, Cylinders: numericas (motor).
- Fuel_Consumption_City_L100km, _Hwy_L100km, _Comb_L100km, _Comb_mpg: numericas
  continuas de consumo.
- CO2_Emissions_gkm: numerica continua, o ALVO declarado pela fonte do dataset.
"""

print("\ndados ausentes e duplicados\n".upper())
missing_data_report(df)

print("\nmedidas de tendencia central e dispersao\n".upper())
numeric_columns = [
    "Engine_Size_L", "Cylinders",
    "Fuel_Consumption_City_L100km", "Fuel_Consumption_Hwy_L100km",
    "Fuel_Consumption_Comb_L100km", "Fuel_Consumption_Comb_mpg",
    "CO2_Emissions_gkm",
]
print(numeric_summary(df, numeric_columns).round(2))
print(f"Model_Year: valor unico = {df['Model_Year'].unique().tolist()} (variancia zero, por isso ficou fora da tabela acima)")

print("\nfoco na variavel alvo: co2_emissions_gkm\n".upper())
target = df["CO2_Emissions_gkm"]
q1, q3 = target.quantile(0.25), target.quantile(0.75)
print(f"Media:          {target.mean():.2f} g/km")
print(f"Mediana:        {target.median():.2f} g/km")
print(f"Moda:           {target.mode().iloc[0]:.2f} g/km (contagem: {target.value_counts().iloc[0]})")
print(f"Desvio padrao:  {target.std():.2f}")
print(f"Variancia:      {target.var():.2f}")
print(f"Q1 / Q3:        {q1:.2f} / {q3:.2f}")
print(f"IQR:            {q3 - q1:.2f}")
print(f"Assimetria:     {target.skew():.2f}")

print("\nboxplots e deteccao de outliers (iqr)\n".upper())
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, column in zip(axes.flat, numeric_columns):
    seaborn.boxplot(y=df[column], ax=ax, color="steelblue")
    ax.set_title(column, fontsize=9)
    ax.set_ylabel("")
for ax in axes.flat[len(numeric_columns):]:
    ax.axis("off")
fig.suptitle("Boxplots das variaveis numericas (Fuel Consumption & CO2 Emissions)")
save_figure(OUTPUT_DIR, "boxplots_variaveis_numericas.png")

print(outlier_report(df, numeric_columns))
"""
Existem alguns outliers superiores em Engine_Size_L, Cylinders, consumo de
combustivel e CO2_Emissions -- veiculos de alta performance/grande porte (V10,
V12, picapes grandes) puxando a cauda direita das distribuicoes. Assim como no
dataset de transito urbano, esses outliers sao carros reais e validos, nao
erros de medicao: sao o extremo esperado de uma frota heterogenea de veiculos.
"""

print("\ncorrelacao entre variaveis e multicolinearidade (heatmap)\n".upper())
correlation = df[numeric_columns].corr()
plt.figure(figsize=(9, 7))
seaborn.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlacao - Fuel Consumption & CO2 Emissions")
save_figure(OUTPUT_DIR, "heatmap_correlacao.png")

corr_with_target = correlation["CO2_Emissions_gkm"].drop("CO2_Emissions_gkm").sort_values()
plt.figure(figsize=(8, 5))
colors = ["crimson" if v < 0 else "seagreen" for v in corr_with_target]
plt.barh(corr_with_target.index, corr_with_target.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Correlacao de cada variavel com CO2_Emissions_gkm")
plt.xlabel("Coeficiente de correlacao (Pearson)")
save_figure(OUTPUT_DIR, "ranking_correlacao_alvo.png")
print(corr_with_target.sort_values(ascending=False))
"""
Diferente do Climate Insights Dataset (correlacoes proximas de zero) e do
transito urbano (correlacoes fracas/moderadas), aqui temos correlacoes fortes
e genuinas: consumo combinado (L/100km) correlaciona 0.98 com CO2_Emissions, e
consumo combinado em mpg correlaciona -0.92 (faz sentido: mpg e o inverso de
consumo). Note tambem que City, Hwy e Comb (L/100km) sao fortemente
correlacionadas ENTRE SI (multicolinearidade) -- em um modelo de regressao que
usasse as tres ao mesmo tempo, os coeficientes individuais ficariam instaveis
e dificeis de interpretar, mesmo o modelo como um todo sendo preciso. Isso e
uma das razoes pelas quais analise multivariada exige checar correlacao entre
os PREDITORES, nao so entre cada preditor e o alvo.
"""

print("\ngraficos de dispersao (scatter / pairplot)\n".upper())
# Escolhida manualmente (em vez do top-3 por |correlacao|) para evitar mostrar
# 3 variaveis de consumo quase identicas entre si (ja discutido como
# multicolinearidade acima) e trazer diversidade: 1 variavel de consumo +
# as 2 variaveis de motor.
top3 = ["Fuel_Consumption_Comb_L100km", "Engine_Size_L", "Cylinders"]
pairplot_columns = ["CO2_Emissions_gkm"] + top3
print("Colunas selecionadas para o pairplot:", pairplot_columns)

seaborn.pairplot(df[pairplot_columns], diag_kind="kde", corner=True)
plt.suptitle("Pairplot: CO2 Emissions x principais variaveis correlacionadas", y=1.02)
save_figure(OUTPUT_DIR, "pairplot_top3.png")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, column in zip(axes, top3):
    seaborn.regplot(data=df, x=column, y="CO2_Emissions_gkm", ax=ax,
                     scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
    ax.set_title(f"{column} vs. CO2_Emissions_gkm")
save_figure(OUTPUT_DIR, "scatter_top3.png")

print("\nanalise categorica: combustivel, transmissao e fabricante\n".upper())
plt.figure(figsize=(9, 5))
seaborn.boxplot(data=df, x="Fuel_Type", y="CO2_Emissions_gkm", hue="Fuel_Type", palette="crest", legend=False)
plt.title("CO2 Emissions por Tipo de Combustivel")
plt.xticks(rotation=15)
save_figure(OUTPUT_DIR, "boxplot_co2_por_combustivel.png")

plt.figure(figsize=(8, 5))
seaborn.boxplot(data=df, x="Transmission_Type", y="CO2_Emissions_gkm", hue="Transmission_Type", palette="mako", legend=False)
plt.title("CO2 Emissions por Tipo de Transmissao")
save_figure(OUTPUT_DIR, "boxplot_co2_por_transmissao.png")

top_makes = df["Make"].value_counts().head(10)
plt.figure(figsize=(9, 6))
seaborn.barplot(x=top_makes.values, y=top_makes.index, hue=top_makes.index, palette="viridis", legend=False)
plt.title("Top 10 fabricantes por numero de modelos no dataset")
plt.xlabel("Numero de modelos")
save_figure(OUTPUT_DIR, "top_fabricantes.png")
"""
Motores a diesel e veiculos de combustivel comum (gasolina regular) mostram
padroes de emissao bem diferentes dos movidos a gas natural/etanol -- ilustra
como uma variavel categorica pode explicar parte da dispersao vista nos
boxplots gerais de CO2_Emissions. Transmissoes manuais tendem a ter emissao
ligeiramente menor que automaticas nesta amostra, mas a diferenca e pequena
perto da variacao explicada pelo motor/consumo.
"""

print("\nresumo das features\n".upper())
print("""
Grupo identificador/categorico: Make (34), Model (351, quase unico), Vehicle_Class (14),
                                 Fuel_Type (5, decodificada), Transmission_Type (4, agrupada)
Grupo do motor:                 Engine_Size_L, Cylinders
Grupo de consumo:                Fuel_Consumption_City/Hwy/Comb_L100km, Fuel_Consumption_Comb_mpg
Alvo:                           CO2_Emissions_gkm (forte correlacao com consumo e motor)
Constante (sem uso analitico):  Model_Year (unico valor = 2001 nesta fatia do dataset)
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
- Neste dataset os outliers (motores grandes, picapes, esportivos) sao carros
  reais e legitimos: removê-los cegamente enviesaria o modelo para subestimar
  a emissao de veiculos de alta performance, justamente os que mais poluem.

Dados ausentes:
- Se a ausencia nao for aleatoria, descartar linhas (dropna) enviesa a amostra.
- Reduzem o tamanho amostral e o poder estatistico disponivel para estimar
  correlacoes e ajustar modelos, especialmente em datasets pequenos como este
  (679 linhas).
- Quando colunas diferentes tem ausencias em linhas diferentes, o calculo de
  correlacao par-a-par (pairwise deletion, padrao do pandas .corr()) usa
  subconjuntos distintos de linhas por par de colunas, podendo gerar matrizes
  de correlacao inconsistentes -- o que quebra tecnicas como PCA.
- Imputar com media/mediana reduz artificialmente a variancia da coluna, o que
  tambem distorce desvio padrao e correlacoes.

Este dataset trouxe um terceiro problema, tao relevante quanto outliers e dados
ausentes: corrupcao de ESQUEMA (nomes de coluna duplicados/ausentes na fonte
original, texto com aspas literais). Sem a limpeza feita no inicio do script, a
segunda coluna "MODEL" teria sido silenciosamente sobrescrita ou renomeada de
forma pouco clara ("MODEL.1"), e as 3 colunas de consumo sem cabecalho
("Unnamed:_9/10/11") teriam ficado impossiveis de interpretar corretamente --
um erro de leitura desse tipo pode levar a usar a variavel errada em uma
analise multivariada (por exemplo, confundir consumo combinado com consumo
rodoviario), com conclusoes erradas mesmo sem nenhum valor ausente ou outlier.
""")

print("\ngerando relatorio html\n".upper())
build_html_report(OUTPUT_DIR, title="Fuel Consumption & CO2 Emissions by Vehicles - EDA")
