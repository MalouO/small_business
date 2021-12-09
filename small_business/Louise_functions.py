import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import googlemaps
from datetime import datetime
import os
import folium
import base64


def latitude(address):
    key = pd.read_csv("raw_data/api_key_vb.csv", header=None)
    key = key.loc[0][0]
    gmaps = googlemaps.Client(key=key)
    geocode_result = gmaps.geocode(address)
    return float(geocode_result[0]['geometry']['location']['lat'])


def longitude(address):
    key = pd.read_csv("raw_data/api_key_vb.csv", header=None)
    key = key.loc[0][0]
    gmaps = googlemaps.Client(key=key)
    # Geocoding an address
    geocode_result = gmaps.geocode(address)
    return float(geocode_result[0]['geometry']['location']['lng'])

## EVOLUTION OF THE REVIEW SCORE


def review_evolution(reviews,
                     type_of_food=0,
                     neighborhood=0,
                     lower_price=0,
                     higher_price=5,
                     restaurant=0):
    reviews = reviews[reviews['price'] > lower_price]
    reviews = reviews[reviews['price'] <= higher_price]

    if restaurant == 0:
        reviews = reviews
    else:
        reviews = reviews[reviews['restaurant_name'] == restaurant]

    if type_of_food == 0:
        reviews = reviews
    else:
        reviews = reviews[reviews['type'] == type_of_food]

    if neighborhood == 0:
        reviews = reviews
    else:
        reviews = reviews[reviews['neighborhood'] == neighborhood]

    reviews_to_plot = reviews.groupby('year').mean()
    increase = reviews_to_plot['comment_ratings'].values[-1] - reviews_to_plot[
        'comment_ratings'].values[0]
    fig, ax = plt.subplots()
    ax.plot(reviews_to_plot['comment_ratings'])

    return fig, increase


#review_evolution(neighborhood='Graça', type_of_food='pizza')
### MAPPING
def number_of_restaurant(data, type_of_food=None, neighborhood=None):
    your_data=data.copy()
    if neighborhood == [] and type_of_food == []:
        your_data = your_data

    if type_of_food != [] and neighborhood == []:
        your_data = your_data[your_data['type'].isin(type_of_food)]

    if neighborhood != []  and type_of_food == []:
        your_data = your_data[your_data['neighborhood'].isin(neighborhood)]

    if type_of_food != []  and neighborhood != [] :
        your_data = your_data[your_data['type'].isin(type_of_food)]
        your_data = your_data[your_data['neighborhood'].isin(neighborhood)]
    return len(your_data.name.unique())


def plot_map(data,
             address=0,
             lower_price=0,
             higher_price=5,
             lowest_rating=0,
             best_rating=5,
             type_of_food=None,
             neighborhood=None,
             latitude2=0,
             longitude2=0):
    your_data = data.copy()
    your_data = data[data['price'] > lower_price]
    your_data = your_data[your_data['price'] <= higher_price]
    your_data = your_data[your_data['rating'] >= lowest_rating]
    your_data = your_data[your_data['rating'] <= best_rating]

    if neighborhood is None and type_of_food is None:
        your_data = your_data

    if type_of_food is not None and neighborhood is None:
        your_data = your_data[your_data['type'].isin(type_of_food)]

    if neighborhood is not None and type_of_food is None:
        your_data = your_data[your_data['neighborhood'].isin(neighborhood)]

    if type_of_food is not None and neighborhood is not None:
        your_data = your_data[your_data['type'].isin(type_of_food)]
        your_data = your_data[your_data['neighborhood'].isin(neighborhood)]

    data_use = your_data

    latlonname = zip(data_use['latitude'], data_use['longitude'],
                     data_use['name'], data_use['price'], data_use['rating'],
                     data_use['type'])

    m = folium.Map(location=[38.72, -9.1385],
                   tiles='CartoDB positron',
                   zoom_start=14)

    icon_size = (50, 50)
    locred = 'small_business/data/locred.png'
    locblue = 'small_business/data/locblue.png'

    if address == 0:
        your_data = your_data
    else:
        longitude1 = longitude(address)
        latitude1 = latitude(address)
        folium.Marker(location=[latitude1, longitude1],
                      tooltip='Current address',
                      icon=folium.CustomIcon(locred,
                                                icon_size=icon_size)).add_to(m)

    for coord in latlonname:
        folium.Marker(
            location=[coord[0], coord[1]],
            tooltip=coord[2],
            popup=
            f"{coord[2]}\nPrice ($): {coord[3]}\nRating:{coord[4]}\nType:{coord[5]}",
            icon=folium.CustomIcon(locblue, icon_size=icon_size)).add_to(m)

    if latitude == 0:
        your_data = your_data
    else:
        longitude2 = longitude2
        latitude2 = latitude2
        folium.Marker(location=[latitude2, longitude2],
                      tooltip='Current address',
                      icon=folium.CustomIcon(locred,
                                             icon_size=icon_size)).add_to(m)

    return m
## RECOMMENDATION IF WE KEEP THE TYPE:

def best_neigh(data, type_of_food=0):
    if type_of_food == []:
        data = data
    else:
        data = data[data['type'].isin(type_of_food)]
    a = data.groupby('neighborhood').mean()
    return (pd.DataFrame(
        a.sort_values('rating', ascending=False).rating.head(3)))

def worse_neigh(data, type_of_food=0):
    #data = pd.read_csv('../small_business/data/restaurants.csv')
    if type_of_food == []:
        data = data
    else:
        data = data[data['type'].isin(type_of_food)]
    a = data.groupby('neighborhood').mean()
    return (pd.DataFrame(a.sort_values('rating',
                                             ascending=True).rating.head(3)))




## RECOMMENDATION IF WE KEEP THE NEIGHBORHOOD:
def best_type(data, neighborhood=0):
    if neighborhood == []:
        data = data
    else:
        data = data[data['neighborhood'].isin(neighborhood)]
    a = data.groupby('type').mean()
    return (pd.DataFrame(a.sort_values('rating',
                                             ascending=False).rating.head(3)))

def worse_type(data, neighborhood=0):
    if neighborhood == []:
        data = data
    else:
        #data = data[data['neighborhood'] == neighborhood]
        data = data[data['neighborhood'].isin(neighborhood)]
    a = data.groupby('type').mean()
    return (pd.DataFrame(a.sort_values('rating',
                                             ascending=True).rating.head(3)))

def best_price_range_neig(data, neighborhood=0):
    if neighborhood == []:
        data = data
    else:
        data = data[data['neighborhood'].isin(neighborhood)]
        data = data[data['rating'] > 4.5]
        data.price.mean()
    return data.price.mean()


def best_price_range(data, type_of_food=0):
    #data = pd.read_csv('../small_business/data/restaurants.csv')
    if type_of_food == []:
        data = data
    else:
        data = data[data['type'].isin(type_of_food)]
        data = data[data['rating'] > 4.5]
        data.price.mean()
    return data.price.mean()


def all_types(data):
    return ('brunch', 'cafe', 'mediterranean', 'european', 'chicken', 'bar',
            'seafood', 'south_america', 'veggie_healthy', 'bistro', 'grill',
            'indian_nepalese', 'portuguese', 'pizza', 'japanese', 'pasta',
            'mexican', 'africa_me', 'burger', 'italian', 'asian', 'pastry',
            'fast_food', 'fado')


def all_neigh(data):
    return ( 'Graça', 'Prazeres', 'Santa Engrácia', 'São Paulo', 'Lapa',
     'Santa Catarina', 'Alto do Pina', 'Encarnação', 'Santa Isabel', 'Socorro',
     'Anjos', 'Santo Estevão', 'Pena', 'Madalena', 'Santos-o-Velho',
     'São Vicente de Fora', 'Mercês', 'Coração de Jesus', 'São José',
     'Sacramento', 'São Miguel', 'Mártires', 'São Cristóvão', 'São Mamede',
     'Sé', 'São Nicolau', 'Campolide', 'Santa Justa', 'São Jorge de Arroios',
     'Santiago', 'Santo Condestável', 'Castelo', 'Alvalade',
     'São Domingos de Benfica', 'Nossa Senhora de Fátima', 'Almargem do Bispo',
     'São Sebastião da Pedreira', 'Santa Maria dos Olivais')


def get_restau(data, name_of_restaurant):
    return data[data['name'] == name_of_restaurant]


def load_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return encoded


def image_tag(path):
    encoded = load_image(path)
    tag = f'<img src="data:image/png;base64,{encoded}">'
    return tag


def background_image_style(path):
    encoded = load_image(path)
    style = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    '''
    return style
