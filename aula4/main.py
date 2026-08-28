import pandas
import numpy

primary_genre = []
is_success = []

def convert_currency(data):
    if pandas.isna(data):
        return numpy.nan
    
    data = str(data).upper()
    DOLLAR_VALUE= 5
    if data.strip().startswith("R$"):
        return (pandas.to_numeric(
            data.replace("R$", "")
                .replace(".", "")
                .replace(",", "")
        ) / 100) * DOLLAR_VALUE
        
    if data.strip().startswith("$"):
        return pandas.to_numeric(
            data.replace("$", "")
                .replace(".", "")
                .replace(",", "")
        ) / 100
    
    if "FREE" in data.upper():
        return 0.00
    

def genre_normalization(data):
    data = str(data).replace(" / ", ", ").replace("/", ", ").replace("; ", ", ").replace(";", ", ").strip()
    primary_genre.append(data.split(",")[0].strip())
    return data

def review_normalization(review):
    if review < 0:
        return 0
    return review

def approval_ratio(df):
    total_reviews = df['Positive_Reviews'] + df['Negative_Reviews']
    return numpy.where(
        total_reviews > 0, df['Positive_Reviews'] / total_reviews, 0.0
    )

def is_success(df):
    condicao = (df['Approval_Ratio'] >= 0.75) & (df['Positive_Reviews'] >= 100)
    return numpy.where(condicao, 1, 0)

df = pandas.read_csv("class_materials/games_data.csv")
print("TYPE CHECK\n")
print(df.dtypes)
print("\n")

df["Price_USD"] = df["Price_USD"].apply(convert_currency)
print("currency conversion\n".upper())
print(df["Price_USD"].head(5))
print("\n")

print("Handling Multiple Categories\n".upper())
df["Genres"] = df["Genres"].apply(genre_normalization)
df["Primary_Genre"] = numpy.array(primary_genre)
print(df[["Genres", "Primary_Genre"]].head(5))
print("\n")

df["Release_Date"] = pandas.to_datetime(df["Release_Date"], format='mixed', errors='coerce')
print("date normalization\n".upper())
print(df["Release_Date"].head(5))

df["Negative_Reviews"] = df["Negative_Reviews"].apply(review_normalization)
df["Positive_Reviews"] = df["Positive_Reviews"].apply(review_normalization)
print("review normalization\n".upper())
print(df[["Negative_Reviews", "Positive_Reviews"]].head(5))

df["Approval_Ratio"] = approval_ratio(df)
print("engajamento relativo\n".upper())
print(df["Approval_Ratio"].head(5))

df['Is_Success'] = is_success(df)
print("variável binária para classificação\n".upper())
print(df['Is_Success'].head(5))