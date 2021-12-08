#import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from m_get_restaurants_data import restaurant_data

X, y = restaurant_data()

def target_X(type_of_food, price, neighborhood, takeaway, dine_in, delivery):
    new_df=pd.DataFrame(columns=['type','price','latitude','longitude', 'dine_in', 'takeaway','delivery', 'drive_through', 'curb_pickup', 'neighborhood'])
    new_row = {'type':type_of_food, 'price': price, 'latitude':'0','longitude':'0', 'takeaway':1,'dine_in':1,'delivery':1, 'drive_through':1, 'curb_pickup':1, 'neighborhood': neighborhood}
    X_user= new_df.append(new_row, ignore_index=True)
    coordinate_transformer = SimpleImputer(strategy="most_frequent")
    price_transformer = SimpleImputer(strategy="mean")
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    preproc = make_column_transformer((price_transformer, ['price']),
                                       (coordinate_transformer, ['latitude, longitude']),
                                        (cat_transformer, ['neighborhood', 'type']), remainder='passthrough')
    pipe = make_pipeline(preproc)
    target = pipe.transform(X_user)
    return target


# target X function not yet finalised, name, and other factors to be seen

def model(X, y, target):
    # split data
    y_class=pd.cut(x=y, bins=[0,4, 5],
                        labels=[ "below_average", "above_average"])
    #instantiate pipeline
    price_transformer = SimpleImputer(strategy="most_frequent")
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    preproc = make_column_transformer((price_transformer, ['price']),
                                        (cat_transformer, ['neighborhood', 'type']), remainder='passthrough')
    pipe = make_pipeline(preproc)
    knn = KNeighborsClassifier(n_neighbors = 5,weights= 'uniform')
    knn.fit(pipe.fit_transform(X),y_class)
    y_probxgb = knn.predict_proba(pipe.transform(target))
    if y_probxgb< 0.8:
            return 'below_average'
    else:
            'above average'
    return knn.kneighbors(target, n_neighbors=10, return_distance=False)

model(X, y, target_X)
