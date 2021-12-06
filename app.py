from jinja2.loaders import PackageLoader
import streamlit as st
import pandas as pd
from small_business.Louise_functions import *
from small_business import classification_model
import subprocess
from streamlit_folium import folium_static

#importing datasets
data = pd.read_csv('small_business/data/restaurants.csv')
reviews = pd.read_csv('raw_data/reviews.csv')

Choice = st.radio(
    'Select your current situation:',
    ('I want to open a restaurant', 'I already have a restaurant'))

st.write(Choice)

if Choice == 'I want to open a restaurant':
    columns = st.columns(4)

    type_of_food = columns[0].selectbox('Type of restaurant',
                                        all_types(data))

    neighborhood = columns[1].selectbox('Select the neighborhood',
                                        all_neigh(data))

    price = columns[2].number_input('Insert a price range')

    address = columns[3].text_input('Address', 'you address')
    #st.write('You selected:', option)

    st.write("Your predicted rating score is:")
    # Mapping
    st.write("Have a look at the other", type_of_food, 'restaurant in',
                    neighborhood)
    folium_static(plot_map(data, type_of_food=type_of_food, neighborhood=neighborhood))

    # Rating evolution
    st.write("Look at the evolution of ", type_of_food, 'restaurant ratings in', neighborhood)
    fig = review_evolution(reviews)
    st.pyplot(fig=fig)

    # Recommendations:
    st.write("The best neighborhoods to open a ", type_of_food, ' restaurant are')
    st.write(best_neigh(data=data, type_of_food=type_of_food))
    #st.write('We recommend a price-range of ',
    #         best_price_range(data=data, type_of_food=type_of_food), 'for', type_of_food)

    st.write("The best type of restaurant to open in ", neighborhood, 'are')
    st.write(best_type(data=data, neighborhood=neighborhood))

    #st.write("The best price range in", neighborhood, 'are')
    #st.write(best_price_range(data=data, type_of_food='pizza'))

else:
    #columns = st.columns(3)

    name_of_restaurant = st.text_input('Name of your restaurant',
                                                 'Dear Breakfast')

    type_of_food=get_restau(data=data,
                   name_of_restaurant=name_of_restaurant)['type'].values[0]

    neighborhood = get_restau(
        data=data,
        name_of_restaurant=name_of_restaurant)['neighborhood'].values[0]

    rating= get_restau(data=data,
                   name_of_restaurant=name_of_restaurant)['rating'].values[0]

    st.write("Your restaurant is categorized as a", type_of_food,
             " restaurant in", neighborhood)

    st.write(
        "Your current rating is ", rating)


    # Mapping
    st.write("Have a look at the other", type_of_food, 'restaurant in', neighborhood)
    folium_static(plot_map(data, type_of_food=type_of_food, neighborhood=neighborhood))

    # Rating evolution
    st.write("Look at the evolution of your restaurant ratings")
    fig = review_evolution(reviews, restaurant=name_of_restaurant)
    st.pyplot(fig=fig)
