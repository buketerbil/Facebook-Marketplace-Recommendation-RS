from clean_tabular_data import dframe, images_df
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
from numpy import asarray
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


merged_df = dframe.merge(images_df, on='product_id')

image_data_df = pd.DataFrame({'path': glob('/Users/macbook/Downloads/resized_images/*.jpg')})

image_data_df['id'] = image_data_df.path.str.split('/').str[-1].str.replace('.jpg', '', regex=False)

image_data_df = image_data_df.merge(merged_df[['category', 'id']], on='id')

image_data_df['category'] = image_data_df.category.str.split('/').str[0]

def from_path_to_feature(path):
  image = Image.open(path)
  data = asarray(image)
  return data.reshape(-1)


image_data_df['features'] = image_data_df.path.apply(from_path_to_feature)

le = preprocessing.LabelEncoder()

image_data_df['target'] = le.fit_transform(image_data_df.category)

clf = LogisticRegression()

clf.fit(np.vstack(image_data_df.features), image_data_df.target)

np.vstack(image_data_df.features)

clf.predict(np.vstack(image_data_df.features)[:10])

image_data_df.target[:10]