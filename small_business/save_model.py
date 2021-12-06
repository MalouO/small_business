#import streamlit as st
from os import X_OK
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from m_get_restaurants_data import restaurant_data
import joblib
import googlemaps


def latitude(column):
    key = pd.read_csv("../raw_data/api_key_vb.csv", header=None)
    key = key.loc[0,0]
    gmaps = googlemaps.Client(key=key)
    # Geocoding an address
    geocode_result = gmaps.geocode(column)
    return float(geocode_result[0]['geometry']['location']['lat'])

def longitude(column):
    key = pd.read_csv("../raw_data/api_key_vb.csv", header=None)
    key = key.loc[0,0]
    gmaps = googlemaps.Client(key=key)
    # Geocoding an address
    geocode_result = gmaps.geocode(column)
    return float(geocode_result[0]['geometry']['location']['lng'])

def get_fitted_pipe(X, y):
    coordinate_transformer = SimpleImputer(strategy="most_frequent")
    price_transformer = SimpleImputer(strategy="mean")
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    preproc = make_column_transformer((price_transformer, ['price']),
                                       (coordinate_transformer, ['latitude', 'longitude']),
                                        (cat_transformer, ['neighborhood', 'type']), remainder='passthrough')
    pipe = make_pipeline(preproc)
    pipe.fit_transform(X) #y_class)
    return pipe

def target_X(type_of_food, price, neighborhood, takeaway, latitude, longitude):
    new_df=pd.DataFrame(columns=['type','price','latitude','longitude', 'dine_in', 'takeaway','delivery', 'neighborhood'])
    new_row = {'type':type_of_food, 'price': price, 'latitude':latitude,'longitude':longitude,'takeaway':takeaway,'dine_in':1,'delivery':1,  'neighborhood': neighborhood}
    X_user= new_df.append(new_row, ignore_index=True)
    return X_user

def build_y(y):
    y_class=pd.cut(x=y, bins=[0,4, 5],
                        labels=[ "below_average", "above_average"])
    return y_class
#target_X(type_of_food = 'brunch', price = '2', neighborhood = 'Graça', takeaway = 1, latitude =38.714376, longitude = -9.130176)
# target X function not yet finalised, name, and other factors to be seen

def train_model(pipeline, X, y_class):

    knn = KNeighborsClassifier(n_neighbors = 5,weights= 'uniform')
    knn.fit(pipeline.fit_transform(X), y_class)
    filename = 'knn_model.joblib'
    joblib.dump(knn, filename)



def predict(X_user_transformed):

    loaded_model = joblib.load('knn_model.joblib')
    y_probxgb = loaded_model.predict_proba(X_user_transformed)
    if y_probxgb[0][0]< 0.8:
            pred =  'below_average'
    else:
            pred = 'above average'
    neighbors =  loaded_model.kneighbors(X_user_transformed, n_neighbors=10, return_distance=False)
    return neighbors, pred



if __name__ == '__main__':
    X, y = restaurant_data()
    pipeline = get_fitted_pipe(X, y)
    X_user = target_X(type_of_food = 'brunch', price = '2', neighborhood = 'Graça', takeaway = 1, latitude =38.714376, longitude = -9.130176)
    X_user_transformed = pipeline.transform(X_user)
    y_class = build_y(y)
    train_model(pipeline, X, y_class)
    predict(X_user_transformed)
