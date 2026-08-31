"""
Link to dataset https://www.openml.org/search?type=data&status=active&id=42896
Sao Paulo in Brazil from December 14, 2009 to December 18, 2009 (From Monday to Friday).
Registered from 7:00 to 20:00 every 30 minutes. 
"""

import pandas
import sklearn.datasets as skdatasets

print("Fetching dataset...")
df: pandas.DataFrame = skdatasets.fetch_openml(data_id=42896, as_frame=True).data
print("Dataset fetched!")

print(df.info())
print(df.head(10))
