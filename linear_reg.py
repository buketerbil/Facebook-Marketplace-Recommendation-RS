import pandas as pd
from clean_tabular_data import dframe
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression



def Simple_LR(data):

    productname_vectorizer = TfidfVectorizer()
    vectorised_product_names = productname_vectorizer.fit_transform(data.product_name).todense()
    data['vectorised product names'] = vectorised_product_names.tolist()

    productdescr_vectorizer = TfidfVectorizer()
    vectorised_product_description = productdescr_vectorizer.fit_transform(data.product_description).todense()
    data['vectorised product descriptions'] = vectorised_product_description.tolist()

    location_vectorizer = TfidfVectorizer()
    vectorised_location = location_vectorizer.fit_transform(data.location).todense()
    data['vectorised location'] = vectorised_location.tolist()
    
    X = data['vectorised product names'] + data['vectorised product descriptions'] + data['vectorised location']
    Y = data['price'].str.replace('£', '').str.replace(',', '').astype(float).values
    X = np.array(X.values.tolist())

    del data
    del vectorised_product_description
    del vectorised_location
    del vectorised_product_names

    regres = LinearRegression().fit(X, Y)

    return print(regres.predict(X[:10]))



outcome = Simple_LR(dframe)
outcome
