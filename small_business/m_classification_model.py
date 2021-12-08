import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from m_get_restaurants_data import restaurant_data

X, y = restaurant_data()

def model(X, y, target):
    # split data

    y_class=pd.cut(x=y, bins=[0,4, 5],
                        labels=[ "below_average", "above_average"])
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.30, random_state=42)

    #instantiate pipeline
    price_transformer = SimpleImputer(strategy="most_frequent")
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    preproc = make_column_transformer((price_transformer, ['price']),
                                        (cat_transformer, ['neighborhood', 'type']), remainder='passthrough')
    pipe = make_pipeline(preproc)
    knn = KNeighborsClassifier(n_neighbors = 5,weights= 'uniform')
    knn.fit(pipe.fit_transform(X_train),y_train)
    y_probxgb = knn.predict_proba(pipe.transform(target))
    if y_probxgb< 0.8:
            return 'below_average'
    else:
            'above average'

model(X, y)
