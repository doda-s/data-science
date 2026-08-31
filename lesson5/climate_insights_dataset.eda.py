"""
Link to dataset https://www.openml.org/search?type=data&status=active&id=46729
"""

import pandas
import sklearn.datasets as skdatasets

print("Fetching dataset...")
df: pandas.DataFrame = skdatasets.fetch_openml(data_id=46729, as_frame=True).data
print("Dataset fetched!")

print(df.info())
print(df.head(10))