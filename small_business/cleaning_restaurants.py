#IMPORTS

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import seaborn as sns
import googlemaps
from datetime import datetime
from sklearn.model_selection import train_test_split

#FUNCTIONS

def keep_numeric(x):
    return re.sub("[^0-9]", "", x)

def replace_price(x):
    if x == "£":
        return 1
    if x == "££":
        return 2
    if x == "£££":
        return 3
    if x == "££££":
        return 4

#def latitude(column):
    #key = pd.read_csv("../raw_data/api_key.csv", header=None)
    #key = key.loc[0,0]
    #gmaps = googlemaps.Client(key=key)
    # Geocoding an address
    #geocode_result = gmaps.geocode(column)
    #return float(geocode_result[0]['geometry']['location']['lat'])

#def longitude(column):
    #key = pd.read_csv("../raw_data/api_key.csv", header=None)
    #key = key.loc[0,0]
    #gmaps = googlemaps.Client(key=key)
    # Geocoding an address
    #geocode_result = gmaps.geocode(column)
    #return float(geocode_result[0]['geometry']['location']['lng'])

def get_postal_code(x):
    return re.findall(r'\d{4}-\d{3}', x)[0]

#Function labels
def labels_eat_on_site (x):
    if 'dine-in' in x:
        return 1
    else:
        return 0

def labels_delivery (x):
    if ('delivery' in x) or ('no-contact delivery' in x):
        return 1
    else:
        return 0

def labels_takeaway (x):
    if ('takeaway' in x):
        return 1
    else:
        return 0

def labels_drive_thru (x):
    if ('drive-through' in x):
        return 1
    else:
        return 0

def labels_no_del(x):
    if ( 'no delivery' in (x)):
        return 1
    else:
        return 0

def labels_curbside(x):
    if ( 'curbside pickup' in (x)):
        return 1
    else:
        return 0

#post code format for post codes table
def post_code2(x):
    return (x[0:4]+'-'+x[4:])


def preproc():
    # Importing the data in form 2 and basic preprocessing (this takes a bit of time)
    data= pd.read_csv("../raw_data/restaurants_clean.csv")
    data=data.rename(columns={0:"Index", 'restaurant_name':"name" , 'restaurant_category':"type", 'restaurant_description':"description", 'restaurant_rating':"rating", 'restaurant_comment_number':"review_count", 'restaurant_price_range': "price", 'restaurant_location':"address", 'restaurant_services':"label"})
    data=data.drop(columns=['Index', 'Unnamed: 0']) # Check whether to keep it or not
    data=data.dropna(subset=['rating']) # dropping rows with no rating
    data.label=data.label.fillna(value='Dine-in')# replacing rows with no label by ['Dine-in']
    data.type=data.type=data.label.fillna(value='Restaurant')

    # Column rating
    data.rating=pd.to_numeric(data.rating, downcast="float")

    # Column review_count
    data.review_count=data.review_count.map(keep_numeric)
    data.review_count=pd.to_numeric(data.review_count, downcast="float")

    #Column price
    data.price = data.price.map(replace_price)

    #Adress into latitude and longitude
    #data['latitude']=data.address.map(latitude)
    #data['longitude']=data.address.map(longitude)

    # handling the label
    data.label= data['label'].str.lower()
    data['dine_in']=data.label.map(labels_eat_on_site)
    data['takeaway']=data.label.map(labels_takeaway)
    data['delivery']=data.label.map(labels_delivery)
    data['drive_through']=data.label.map(labels_drive_thru)
    data['no_del_exp']=data.label.map(labels_no_del)
    data['curb_pickup']=data.label.map(labels_curbside)

    # handling the type
    data.type=data.type.replace('restaurant', '', regex=True)
    data.type= data.type.str.lower()

    #add postal code
    data['postal_code']=data.address.map(get_postal_code)

    #handling remaining null values (price):
    data.price=data.price.fillna(value=np.mean(data.price)).astype(int) ## CHOICE 1 TO BE CONFIRMED: FILLING THE NA ON PRICE WITH AVERAGE
    print('done')
    return data


def preproc_postal():
    data_post=pd.read_csv("../raw_data/cod_post_freg_matched.txt")
    data_post.CodigoPostal=data_post.CodigoPostal.astype(str)
    data_post.CodigoPostal=data_post.CodigoPostal.map(post_code2)
    data_post=data_post[['CodigoPostal', 'Concelho', 'Freguesia']]
    data_post=data_post.rename(columns={'CodigoPostal':'postal_code', 'Distrito': 'district', 'Concelho': 'municipality', 'Freguesia': 'neighborhood'} )
    return data_post

def merge_post():
    data=preproc().merge(preproc_postal(), on='postal_code', how='left')
    return data

def get_data():
    df = merge_post()
    X = df['']
    y = df['rating']
    X, y, X_test, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X, y, X_test, y_test

#decide X
# whats the bug fix?
