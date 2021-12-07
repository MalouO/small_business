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


def restaurant_data(data):
    #data = pd.read_csv('../small_business/data/restaurants.csv')
    #data = data.drop(columns='Unnamed: 0')
    X = data.drop(columns=[
        'rating', 'name', 'address', 'label', 'curb_pickup', 'drive_through',
        'postal_code', 'no_del_exp', 'municipality', 'review_count'
    ])
    y = data['rating']
    return X, y


def latitude(address):
    key = pd.read_csv("../raw_data/api_key_vb.csv", header=None)
    key = key.address
    gmaps = googlemaps.Client(key=key)
    # Geocoding an address
    geocode_result = gmaps.geocode(address)
    return float(geocode_result[0]['geometry']['location']['lat'])


def longitude(column):
    key = pd.read_csv("../raw_data/api_key_vb.csv", header=None)
    key = key.address
    gmaps = googlemaps.Client(key=key)
    # Geocoding an address
    geocode_result = gmaps.geocode(column)
    return float(geocode_result[0]['geometry']['location']['lng'])


def get_fitted_pipe(X):
    num_transformer = StandardScaler()
    price_transformer = SimpleImputer(strategy="mean")
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    preproc = make_column_transformer(
        (price_transformer, ['price']),
        (num_transformer, ['latitude', 'longitude']),
        (cat_transformer, ['neighborhood', 'type']),
        remainder='passthrough')
    pipe = make_pipeline(preproc)
    pipe.fit_transform(X)
    return pipe


def target_X(type_of_food, price, neighborhood, latitude, longitude):
    new_df = pd.DataFrame(columns=[
        'type', 'price', 'latitude', 'longitude', 'dine_in', 'takeaway',
        'delivery', 'neighborhood'
    ])
    new_row = {
        'type': type_of_food,
        'price': price,
        'latitude': latitude,
        'longitude': longitude,
        'takeaway': 1,
        'dine_in': 1,
        'delivery': 1,
        'neighborhood': neighborhood
    }
    X_user = new_df.append(new_row, ignore_index=True)
    return X_user


def build_y(y):
    y_class = pd.cut(x=y,
                     bins=[0, 4, 5],
                     labels=["below_average", "above_average"])
    return y_class


def train_model(pipeline, X, y_class):
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(pipeline.fit_transform(X), y_class)
    filename = 'knn_model.joblib'
    joblib.dump(knn, filename)


def predict(X_user_transformed):
    loaded_model = joblib.load('knn_model.joblib')
    y_probxgb = loaded_model.predict_proba(X_user_transformed)
    if y_probxgb[0][0] < 0.8:
        pred = 'below_average'
    else:
        pred = 'above average'
    neighbors = loaded_model.kneighbors(X_user_transformed,
                                        n_neighbors=10,
                                        return_distance=False)
    return neighbors, pred


if __name__ == '__main__':
    X, y = restaurant_data()
    pipeline = get_fitted_pipe(X, y)
    X_user = target_X(type_of_food=type_of_food,
                      price=price,
                      neighborhood=neighborhood,
                      takeaway=1,
                      latitude=38.714376,
                      longitude=-9.130176)
    X_user_transformed = pipeline.transform(X_user)
    y_class = build_y(y)
    train_model(pipeline, X, y_class)
    predict(X_user_transformed)


def output_model(data, type_of_food, price, neighborhood):
    X, y = restaurant_data()
    pipeline = get_fitted_pipe(X, y)
    X_user = target_X(type_of_food=type_of_food,
                      price=price,
                      neighborhood=neighborhood,
                      takeaway=1,
                      latitude=38.714376,
                      longitude=-9.130176)
    X_user_transformed = pipeline.transform(X_user)
    y_class = build_y(y)
    train_model(pipeline, X, y_class)
    return predict(X_user_transformed)
