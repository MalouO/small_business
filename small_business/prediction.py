#import streamlit as st
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import googlemaps

def restaurant_data():

    data = pd.read_csv('../small_business/data/restaurants.csv')
    data = data.drop(columns = 'Unnamed: 0')
    X = data.drop(columns=['rating','name', 'address', 'label', 'curb_pickup', 'drive_through', 'postal_code', 'no_del_exp', 'municipality', 'review_count'])
    y = data['rating']
    return X, y

def latitude(address):
    key = pd.read_csv("../raw_data/api_key_vb.csv", header=None)
    key = key.loc[0][0]
    gmaps = googlemaps.Client(key=key)
    geocode_result = gmaps.geocode(address)
    return float(geocode_result[0]['geometry']['location']['lat'])

def longitude(address):
    key = pd.read_csv("../raw_data/api_key_vb.csv", header=None)
    key = key.loc[0][0]
    gmaps = googlemaps.Client(key=key)
    geocode_result = gmaps.geocode(address)
    return float(geocode_result[0]['geometry']['location']['lng'])

def address_imputer(address, neighborhood, X):
    if address == None:
        all_latitude = pd.DataFrame(X['latitude'].groupby(X['neighborhood']).mean())
        all_longitude = pd.DataFrame(X['longitude'].groupby(X['neighborhood']).mean())
        for place in all_latitude.index:
            if place == neighborhood:
                lattitude_replace = all_latitude.loc[place].values[0]
        for place in all_longitude.index:
            if place == neighborhood:
                longitude_replace = all_longitude.loc[place].values[0]
        return lattitude_replace, longitude_replace
    else:
        longitude1 = longitude(address)
        latitude1 = latitude(address)
        return latitude1, longitude1

def get_fitted_pipe(X):
    num_transformer = StandardScaler()
    price_transformer = SimpleImputer(strategy="mean")
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    preproc = make_column_transformer((price_transformer, ['price']),
                                       (num_transformer, ['latitude', 'longitude']),
                                        (cat_transformer, ['neighborhood', 'type']), remainder='passthrough')
    pipe = make_pipeline(preproc)
    pipe.fit_transform(X)
    return pipe

def target_X(type_of_food, price, neighborhood, latitude, longitude):
    new_df=pd.DataFrame(columns=['type','price','latitude','longitude', 'dine_in', 'takeaway','delivery', 'neighborhood'])
    new_row = {'type':type_of_food, 'price': price, 'latitude':latitude,'longitude':longitude,'takeaway':1,'dine_in':1,'delivery':1,  'neighborhood': neighborhood}
    X_user= new_df.append(new_row, ignore_index=True)
    return X_user

def build_y(y):
    y_class=pd.cut(x=y, bins=[0,4.2, 5],
                        labels=["bad idea!", "good idea!"])
    return y_class

def train_model(pipeline, X, y_class):

    knn = KNeighborsClassifier(n_neighbors = 5,weights= 'uniform')
    knn.fit(pipeline.fit_transform(X), y_class)
    filename = 'knn_model.joblib'
    joblib.dump(knn, filename)


def predict(X_user_transformed):

    loaded_model = joblib.load('knn_model.joblib')
    y_probxgb = loaded_model.predict_proba(X_user_transformed)
    if y_probxgb[0][0]< 0.8:
            pred =  'bad idea!'
    else:
            pred = 'good idea!'
    return pred

def return_predicted_probability(X_user_transformed):
    loaded_model = joblib.load('knn_model.joblib')
    y_probxgb = loaded_model.predict_proba(X_user_transformed)
    return y_probxgb

def all_results(type_of_food, price, neighborhood, latitude, longitude):
        loaded_model = joblib.load('knn_model.joblib')
        y_probxgb = loaded_model.predict_proba(X_user_transformed)


def neighbours(X_user_transformed):
    loaded_model = joblib.load('knn_model.joblib')
    neighbors =  loaded_model.kneighbors(X_user_transformed, n_neighbors=10, return_distance=False)
    return neighbors

if __name__ == '__main__':
    X, y = restaurant_data()
    pipeline = get_fitted_pipe(X, y)
    neighborhood = neighborhood
    address = address
    latitude1, longitude1 = address_imputer(address, neighborhood)
    X_user = target_X(type_of_food = type_of_food, price = price, neighborhood = neighborhood, takeaway = 1, latitude = latitude1, longitude=longitude1)
    X_user_transformed = pipeline.transform(X_user)
    y_class = build_y(y)
    #train_model(pipeline, X, y_class)
    predict(X_user_transformed)
    neighbours(X_user_transformed)

def output_model(data, type_of_food, price, address, neighborhood):
    X, y = restaurant_data()
    pipeline = get_fitted_pipe(X, y)
    neighborhood = neighborhood
    address = address
    latitude, longitude = address_imputer(address, neighborhood)
    X_user = target_X(type_of_food = type_of_food, price = price, neighborhood = neighborhood, takeaway = 1, dine_in =1,'delivery':1, latitude = latitude, longitude=longitude)
    X_user_transformed = pipeline.transform(X_user)
    y_class = build_y(y)
    predict(X_user_transformed)
    neighbours(X_user_transformed)
