import csv
from operator import index
import pandas as pd 

dframe = pd.read_csv("/Users/macbook/Downloads/Products.csv", index_col=0)
images_df = pd.read_csv("/Users/macbook/Downloads/Images.csv", index_col=0)

dframe = dframe.dropna(axis='index', how='any') 

dframe = dframe.rename(columns={'id': 'product_id'})

dframe