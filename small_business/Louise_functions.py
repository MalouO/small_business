import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import googlemaps
from datetime import datetime
import os
import folium

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


def plot_map(data,
             address=0,
             lower_price=0,
             higher_price=5,
             lowest_rating=0,
             best_rating=5,
             type_of_food=0,
             neighborhood=0,
             latitude2=0,
             longitude2=0):
    your_data = data.copy()
    your_data = data[data['price'] > lower_price]
    your_data = your_data[your_data['price'] <= higher_price]
    your_data = your_data[your_data['rating'] >= lowest_rating]
    your_data = your_data[your_data['rating'] <= best_rating]


    if neighborhood == 'all':
        your_data = your_data

    if type_of_food == 0 or type_of_food == 'all':
        your_data = your_data
    else:
        your_data = your_data[your_data['type'] == type_of_food]

    if neighborhood == 0 or neighborhood=='all':
        your_data = your_data
    else:
        your_data = your_data[your_data['neighborhood'] == neighborhood]

    data_use = your_data

    latlonname = zip(data_use['latitude'], data_use['longitude'],
                     data_use['name'], data_use['price'], data_use['rating'],
                     data_use['type'])

    m = folium.Map(location=[38.709223, -9.1383],
                   titles='small businesses',
                   zoom_start=13.45)

    if address == 0:
        your_data = your_data
    else:
        longitude1 = longitude(address)
        latitude1 = latitude(address)
        folium.Marker(location=[latitude1, longitude1],
                      tooltip=str,
                      popup='Your_restaurant',
                      icon=folium.Icon(color='red', icon='home',
                                       prefix='fa')).add_to(m)

    for coord in latlonname:
        folium.Marker(
            location=[coord[0], coord[1]],
            tooltip=str,
            popup=
            f'\n Name: {coord[2]} \n Price ($): {coord[3]} \n Rating:{coord[4]}\n Type:{coord[5]}'
        ).add_to(m)

    if latitude == 0:
        your_data = your_data
    else:
        longitude2 = longitude2
        latitude2 = latitude2
        folium.Marker(location=[latitude2, longitude2],
                      tooltip=str,
                      popup='Your_restaurant',
                      icon=folium.Icon(color='red', icon='home',
                                       prefix='fa')).add_to(m)

    return m

## RECOMMENDATION IF WE KEEP THE TYPE:
def best_neigh(data, type_of_food=0):
    if type_of_food == 0 or type_of_food=='all':
        data = data
    else:
        data = data[data['type'] == type_of_food]
    a = data.groupby('neighborhood').mean()
    return (pd.DataFrame(
        a.sort_values('rating', ascending=False).rating.head(3)))


def worse_neigh(data, type_of_food=0):
    #data = pd.read_csv('../small_business/data/restaurants.csv')
    if type_of_food == 0 or type_of_food == 'all':
        data = data
    else:
        data = data[data['type'] == type_of_food]
    a = data.groupby('neighborhood').mean()
    return (pd.DataFrame(a.sort_values('rating',
                                             ascending=True).rating.head(3)))




## RECOMMENDATION IF WE KEEP THE NEIGHBORHOOD:
def best_type(data, neighborhood=0):
    #data = pd.read_csv('../small_business/data/restaurants.csv')
    if neighborhood == 0 or neighborhood =='all':
        data = data
    else:
        data = data[data['neighborhood'] == neighborhood]
    a = data.groupby('type').mean()
    return (pd.DataFrame(a.sort_values('rating',
                                             ascending=False).rating.head(3)))


def worse_type(data, neighborhood=0):
    #data = pd.read_csv('../small_business/data/restaurants.csv')
    if neighborhood == 0 or neighborhood == 'all':
        data = data
    else:
        #data = data[data['neighborhood'] == neighborhood]
        data = data[data['neighborhood'].isin(neighborhood)]
    a = data.groupby('type').mean()
    return (pd.DataFrame(a.sort_values('rating',
                                             ascending=True).rating.head(3)))

def best_price_range_neig(data, neighborhood=0):
    #data = pd.read_csv('../small_business/data/restaurants.csv')
    if neighborhood == 0:
        data = data
    else:
        data = data[data['neighborhood'] == neighborhood]
        data = data[data['rating'] > 4.5]
        data.price.mean()
    return data.price.mean()


def best_price_range(data, type_of_food=0):
    #data = pd.read_csv('../small_business/data/restaurants.csv')
    if type_of_food == 0:
        data = data
    else:
        data = data[data['type'] == type_of_food]
        data = data[data['rating'] > 4.5]
        data.price.mean()
    return data.price.mean()


def all_types(data):
    return ('all','brunch', 'cafe', 'mediterranean', 'european', 'chicken', 'bar',
            'seafood', 'south_america', 'veggie_healthy', 'bistro', 'grill',
            'indian_nepalese', 'portuguese', 'pizza', 'japanese', 'pasta',
            'mexican', 'africa_me', 'burger', 'italian', 'asian', 'pastry',
            'fast_food', 'fado')


def all_neigh(data):
    return ('all', 'Graça', 'Prazeres', 'Santa Engrácia', 'São Paulo', 'Lapa',
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
