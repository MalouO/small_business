import streamlit as st
import pandas as pd
from small_business.Louise_functions import *
from streamlit_folium import folium_static
from small_business.LC_class_mod import *
from small_business.Victor_functions import *

#Importing datasets
data = pd.read_csv('small_business/data/restaurants.csv')
reviews = pd.read_csv('data/reviews.csv')


#Creating Side Bar
st.sidebar.markdown("# What is your current situation? ")
Choice = st.sidebar.radio(
    'Please, select your current situation:',
    ('I want to open a restaurant 🎈', 'I already have a restaurant'))

#1st case : I want to open a restaurant

if Choice == 'I want to open a restaurant 🎈':
    #Asking the user to chose the features of the restaurant:
    st.title("Help me open a restaurant")
    columns = st.columns(3)
    type_of_food = columns[0].selectbox('Type of restaurant:',
                                        all_types(data))

    neighborhood = columns[1].selectbox('Select the neighborhood:',
                                        all_neigh(data))

    price = columns[2].slider('Insert a price range: (1 for cheap, 4 for expensive)',
        min_value=1,
        max_value=4)

    if st.checkbox('I already know the precise address'):
        address = st.text_input('Address', 'Rua da Graça 45, 1170-168 Lisboa')
    else:
        address= 0

    #Predicting a score for the restaurant:
    if type_of_food != 'all' and neighborhood != 'all':
        st.markdown(f"#### We think that this idea is a:")
        st.markdown(
            output_model(data=data,
                        type_of_food=type_of_food,
                        price=price,
                        address=address,
                        neighborhood=neighborhood)[0])
    else:
        st.markdown(f"##### Please Select only one type of restaurant and one neighborhood to have a prediction! ")


    #Mapping the similar restaurants + the address if the user inputed an address:
    st.markdown(
        f'Have a look a the other *{type_of_food.capitalize()}* restaurants in *{neighborhood.capitalize()}* neighborhood(s):'
    )
    folium_static(plot_map(data=data, type_of_food=type_of_food, neighborhood=neighborhood, address=address))

    # Plot the evolution of ratings
    #st.write("Look at the evolution of ", type_of_food, 'restaurant ratings in', neighborhood)
    #fig = review_evolution(reviews)[0]
    #increase = review_evolution(reviews)[1]

    # Provide feedback on the evolution of rating

    #plot the figure

    # Provide recommendations on the neighborhood or the type of food:

    #st.write("The **best neighborhoods** to open a ",
    #         type_of_food.capitalize(), 'restaurant are:')
    #st.write(best_neigh(data=data, type_of_food=type_of_food))

    st.markdown(f"#### Our recommendations:")

    columns = st.columns(2)

    help_neig = columns[0].button('Help me choose my neighboorhood')
    help_type = columns[1].button('Help me choose my type of restaurant')

    if help_neig:
        st.markdown(f'##### Help me choose my neighboorhood!')
        st.write(
            f"**Best neighborhoods** to open a *{type_of_food.capitalize()}* restaurant:"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "",
            best_neigh(data=data, type_of_food=type_of_food).index[0],
            np.round(best_neigh(data=data, type_of_food=type_of_food).values[0][0],
                    decimals=1))
        col2.metric("",
                    best_neigh(data=data, type_of_food=type_of_food).index[1],
                    np.round(best_neigh(data=data, type_of_food=type_of_food).values[1][0], decimals=1))
        col3.metric(
            "",
            best_neigh(data=data, type_of_food=type_of_food).index[2],
            np.round(best_neigh(data=data, type_of_food=type_of_food).values[2][0],
                    decimals=1))

        st.write(
            f"**Worse neighborhoods** to open a *{type_of_food.capitalize()}* restaurant:"
        )
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "",
            worse_neigh(data=data, type_of_food=type_of_food).index[0], - np.round(
                worse_neigh(data=data, type_of_food=type_of_food).values[0][0],
                decimals=1))
        col2.metric(
            "",
            worse_neigh(data=data, type_of_food=type_of_food).index[1], -
            np.round(worse_neigh(data=data,
                                type_of_food=type_of_food).values[1][0],
                    decimals=1))
        col3.metric(
            "",
            worse_neigh(data=data, type_of_food=type_of_food).index[2],
            - np.round(worse_neigh(data=data,
                                type_of_food=type_of_food).values[2][0],
                    decimals=1))


        st.write(
            f"For {type_of_food.capitalize()} restaurants, we recommend a **price range** of **{np.round(best_price_range(data=data, type_of_food=type_of_food),decimals=1)}**"
        )

    if help_type:
        st.markdown(f'##### Help me choose my type of restaurant!')
        st.write(
            f"The **most successful restaurants** in *{neighborhood.capitalize()}* neighborhood are currently:"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "",
            best_type(data=data, neighborhood=neighborhood).index[0].capitalize(),
            np.round(best_type(data=data,
                               neighborhood=neighborhood).values[0][0],
                     decimals=1))
        col2.metric(
            "",
            best_type(data=data,
                      neighborhood=neighborhood).index[1].capitalize(),
            np.round(best_type(data=data,
                               neighborhood=neighborhood).values[1][0],
                     decimals=1))
        col3.metric(
            "",
            best_type(data=data,
                      neighborhood=neighborhood).index[2].capitalize(),
            np.round(best_type(data=data,
                               neighborhood=neighborhood).values[2][0],
                     decimals=1))

        st.write(
             f"The **least successful restaurants** in *{neighborhood.capitalize()}* neighborhood are currently:")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "",
            worse_type(data=data,
                       neighborhood=neighborhood).index[0].capitalize(),
            -np.round(worse_type(data=data,
                                 neighborhood=neighborhood).values[0][0],
                      decimals=1))
        col2.metric(
            "",
            worse_type(data=data, neighborhood=neighborhood).index[1].capitalize(), -
            np.round(worse_type(data=data,
                                neighborhood=neighborhood).values[1][0],
                    decimals=1))
        col3.metric(
            "",
            worse_type(data=data,
                       neighborhood=neighborhood).index[2].capitalize(),
            -np.round(worse_type(data=data,
                                 neighborhood=neighborhood).values[2][0],
                      decimals=1))


        #st.write(best_type(data=data, neighborhood=neighborhood))

        st.write(
            "The best **price range** in", neighborhood, 'is',
            np.round(best_price_range_neig(data=data, neighborhood=neighborhood),
                    decimals=1))

        #st.write("The best price range in", neighborhood, 'is',


    if neighborhood == 'all':
        neighborhood = [i for i in reviews.neighborhood.unique()]
    else:
        neighborhood=[neighborhood]
    if type_of_food == 'all':
        type_of_food = [i for i in reviews.type.unique()]
    else:
        type_of_food = [type_of_food]

    fig = restaurant_full_analysis(df=reviews,
                                   neighborhood=neighborhood,
                                   type_rest=type_of_food)

    st.markdown(
        "#### Look at the evolution of similar restaurants' ratings and get inspiration for yours! "
    )
    st.write(f'*{type_of_food} restaurants in {neighborhood}*')

    st.pyplot(fig=fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)


#2nd case : I already have a retaurant

else:
    st.title("Help me improve my restaurant")
    #Asking the user to chose the features of its restaurant:
    #columns = st.columns(3)
    name_of_restaurant = st.text_input('Please enter the name of your restaurant',
                                                 'Tamarind')

    #Extraction of the information of the restaurant from our database:
    type_of_food=get_restau(data=data,
                   name_of_restaurant=name_of_restaurant)['type'].values[0]

    neighborhood = get_restau(
        data=data,
        name_of_restaurant=name_of_restaurant)['neighborhood'].values[0]

    rating = get_restau(
        data=data, name_of_restaurant=name_of_restaurant)['rating'].values[0]

    latitude2 = get_restau(
        data=data,
        name_of_restaurant=name_of_restaurant)['latitude'].values[0]

    longitude2 = get_restau(
        data=data, name_of_restaurant=name_of_restaurant)['longitude'].values[0]

    #Show the extracted features of the restaurant:
    st.write(f'Your restaurant is categorized as a *{type_of_food}* restaurant in *{neighborhood}*')

    st.write(
        "Your current rating is ", rating)


    #Mapping the similar restaurants + the current restaurant in red with a home:
    st.write(f'Have a look at your restaurant (in red), but also at the other *{type_of_food.capitalize()}* restaurants in *{neighborhood}* :')
    folium_static(
        plot_map(data,
                 type_of_food=type_of_food,
                 neighborhood=neighborhood,
                 latitude2=latitude2,
                 longitude2=longitude2))

    # Plot the evolution of ratings
    #fig = review_evolution(reviews, restaurant=name_of_restaurant)[0]
    #st.pyplot(fig=fig)

    st.markdown("#### Look at the evolution of your restaurant ratings and understand it ! ")
    fig=restaurant_full_analysis(df=reviews, restaurant=[name_of_restaurant])
    st.pyplot(fig=fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    #if increase >= 0:
    #    st.write("Congrats! On average, ratings of", type_of_food,
    #             "reviews in", neighborhood, 'have increased!')
    #else:
    #    st.write("Unfortunately, look like", type_of_food, "reviews in",
    #             neighborhood, 'have decreased...')
