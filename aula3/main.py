import pandas

def structural_inspection_and_validation(df):
    print(df.dtypes)
    print("\n")
    
    for column in df.columns:
        pass

def data_standardization(df):
    DATE_COLUMNS = [
        'Signup_Date',
        'Last_purchase_date',
    ]
    date_pattern = r'(\d{2}[-/]\d{2}[-/]\d{4})'
    extracted = pandas.DataFrame()
    for column in DATE_COLUMNS:
        print(f"{column}")
        date_series = df[column].astype(str).str.extract(date_pattern, expand=False)
        extracted[f'{column}_Extracted'] = pandas.to_datetime(
            date_series,
            dayfirst=True, 
            errors='coerce',
        )
        print(extracted.dropna())
        print("\n")

df = pandas.read_csv("class_materials/data_table.csv")

structural_inspection_and_validation(df)
data_standardization(df)