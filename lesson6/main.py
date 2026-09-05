import pandas
import numpy

df = pandas.read_json("./class_materials/dados_hospedagem.json")
print(df.head(5))

df = pandas.json_normalize(df["info_moveis"])
print(df)

columns_list = list(df.columns)
print(columns_list)

df = df.explode(columns_list[3:])
print(df)

df.reset_index(drop=True, inplace=True)
print(df)

df["max_hospedes"] = df["max_hospedes"].astype(numpy.int64)
print(df)

columns_numerics = ["quantidade_banheiros", "quantidade_quartos", "quantidade_camas"]
df[columns_numerics] = df[columns_numerics].astype(numpy.int64)
print(df[columns_numerics])

df["avaliacao_geral"] = df["avaliacao_geral"].astype(numpy.float64)
print(df["avaliacao_geral"])

for col in  ["preco", "taxa_deposito", "taxa_limpeza"]:
    df[col] = df[col].apply(
        lambda data:
            pandas.to_numeric(
                data.replace("$", "").replace(",", "").replace(".", "").strip()
            )/100
    )
print(df[["preco", "taxa_deposito", "taxa_limpeza"]])

print(df)