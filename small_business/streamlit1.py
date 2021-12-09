#from jinja2.loaders import PackageLoader
import streamlit as st
import pandas as pd
from small_business.streamlit1 import *
#import subprocess
from streamlit_folium import folium_static
from small_business.streamlit3 import *


#importing datasets
data = pd.read_csv('small_business/data/restaurants.csv')
reviews = pd.read_csv('raw_data/reviews.csv')
address= 0


#Creating Side Bar
st.sidebar.markdown("# What is your current situation? ")
Choice = st.sidebar.radio(
    'Please, select your current situation:',
        ('I want to open a restaurant', 'I already have a restaurant'))


if Choice == 'I want to open a restaurant':
    st.title("Help me open a restaurant")
    columns = st.columns(3)
    type_of_food = columns[0].selectbox('Type of restaurant:',
                                        all_types(data))

    neighborhood = columns[1].selectbox('Select the neighborhood:',
                                        all_neigh(data))

    price = columns[2].slider('Insert a price range:', min_value=1, max_value=4)

    if st.checkbox('I already know the precise address'):
        address = st.text_input('Address', 'Rua da Graça 45, 1170-168 Lisboa')
    else:
        address= 0

    st.markdown("#### We think that this idea is a:")

    st.markdown(
        output_model(data=data,
                     type_of_food=type_of_food,
                     price=price,
                     address=address,
                     neighborhood=neighborhood)[0])


    # Mapping
    st.write("Have a look at the other", type_of_food, 'restaurant in',
                    neighborhood)
    folium_static(plot_map(data=data, type_of_food=type_of_food, neighborhood=neighborhood, address=address))

    # Rating evolution
    st.write("Look at the evolution of ", type_of_food, 'restaurant ratings in', neighborhood)
    fig = review_evolution(reviews)[0]
    increase = review_evolution(reviews)[1]

    if increase >= 0:
        st.write("Congrats! On average, ratings of", type_of_food, "reviews in",
                 neighborhood, 'have increased since 2010!')
    else:
        st.write("Unfortunately, look like", type_of_food, "reviews in",
                 neighborhood, 'have decreased since 2010...')

    st.pyplot(fig=fig)

    # Recommendations:
    columns = st.columns(2)

    st.write("The best neighborhoods to open a ", type_of_food,
                     ' restaurant are')
    st.write(best_neigh(data=data, type_of_food=type_of_food))
    #st.write('We recommend a price-range of ',
    #         best_price_range(data=data, type_of_food=type_of_food), 'for', type_of_food)

    st.write("The best type of restaurant to open in ", neighborhood, 'are')
    st.write(best_type(data=data, neighborhood=neighborhood))

    #st.write("The best price range in", neighborhood, 'are')
    #st.write(best_price_range(data=data, type_of_food='pizza'))

else:
    #columns = st.columns(3)
    st.title("Help me improve my restaurant")
    name_of_restaurant = st.text_input('Please enter the name of your restaurant',
                                                 'Dear Breakfast')

    type_of_food=get_restau(data=data,
                   name_of_restaurant=name_of_restaurant)['type'].values[0]

    neighborhood = get_restau(
        data=data,
        name_of_restaurant=name_of_restaurant)['neighborhood'].values[0]

    rating = get_restau(
        data=data, name_of_restaurant=name_of_restaurant)['rating'].values[0]

    latitude1 = get_restau(
        data=data,
        name_of_restaurant=name_of_restaurant)['latitude'].values[0]

    longitude1 = get_restau(
        data=data, name_of_restaurant=name_of_restaurant)['longitude'].values[0]

    st.write("Your restaurant is categorized as a", type_of_food,
             " restaurant in", neighborhood)

    st.write(
        "Your current rating is ", rating)


    # Mapping
    st.write("Have a look at the other", type_of_food, 'restaurant in', neighborhood)
    folium_static(plot_map(data, type_of_food=type_of_food, neighborhood=neighborhood))

    # Rating evolution
    st.write("Look at the evolution of your restaurant ratings")
    fig = review_evolution(reviews, restaurant=name_of_restaurant)[0]
    st.pyplot(fig=fig)
