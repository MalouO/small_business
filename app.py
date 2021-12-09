from numpy.core.function_base import add_newdoc
import streamlit as st
import pandas as pd
from small_business.Louise_functions import *
from streamlit_folium import folium_static
from small_business.LC_class_mod import *
from small_business.Victor_functions import *
from PIL import Image
import base64
from small_business.malou_functions import *



st.set_page_config(layout='wide')



#Importing datasets
data = pd.read_csv('small_business/data/restaurants.csv')
reviews = pd.read_csv('small_business/data/reviews.csv')
df = pd.read_csv('small_business/data/data_probabilities.csv')
#df= df.drop(columns='unnamed')

#Creating Side Bar
# image_path = f"small_business/data/restaurant_image.jpeg"
# st.sidebar.write(background_image_style(image_path), unsafe_allow_html=True)

st.sidebar.markdown("# What is your current situation? ")

Choice = st.sidebar.radio(
    'Please, select your current situation:',
    ('I want to open a restaurant 🎈', 'I already have a restaurant 🍜'))



#1st case : I want to open a restaurant
if Choice == 'I want to open a restaurant 🎈':
    #Asking the user to chose the features of the restaurant:
    st.title("Help me open a restaurant")
    columns = st.columns(3)

    type_of_food = columns[0].multiselect('Type of restaurant 🍛',
                                          all_types(data),
                                          default=None)

    neighborhood = columns[1].multiselect('Select the neighborhood 📍 ',
                                          all_neigh(data),
                                          default=None)

    price = columns[2].slider(
        'Insert a price range 💵',
        min_value=1,
        max_value=4, help='1 for cheap, 4 for expensive' )
    #help = 'Filter report to show only one hospital'

    #st.output_new(type_of_food=type_of_food, neighborhood=neighborhood, price=price)

    if st.checkbox('I already know the precise address'):
        address = st.text_input('Address', 'Praza de Alegria, Lisbon')
    else:
        address= 0



    # Creating a list for restaurants types and neighborhood:
    if type_of_food == []:
        selected_types=' '
    else:
        selected_types=', '.join([i.capitalize() for i in type_of_food])

    if neighborhood == []:
        selected_neighborhoods = 'all Lisbon'
    else:
        selected_neighborhoods =', '.join([i.capitalize() for i in neighborhood])

#st.write(output_new(type_of_food, neighborhood, [price], df))

#Predicting a score for the restaurant:
    st.markdown(f'#### 💪 **Performance prediction**  ')
    if len(type_of_food)==0 or len(neighborhood)==0:
        st.warning(
            f" Please select at least one neighborhood and one type of food to have a performance prediction"
        )
    elif len(type_of_food)==1 and len(neighborhood)==1:
        st.markdown(
            f"We think that this idea is a  👉 {output_model(data=data, type_of_food=type_of_food[0], price=price, address=address, neighborhood=neighborhood[0])[0]}   👈 "
        )
    else:
        st.write(
            f"We ranked the best combination type of food/ neigbhborhood among your selection:",
            )
        st.write(
            get_selection(type_of_food, neighborhood, price, df),
            help=
            'outputs corresponds to the probability of being classified by the model as a "good idea"'
        )


    #st.markdown(
    #    f"##### **Performance prediction**:  ✋ * Please select only one type of restaurant and one neighborhood to have a prediction* ✋"
    #)

    st.markdown(
        f'#### 📍 Have a look at the other {selected_types} restaurant(s) in {selected_neighborhoods} neighborhood(s)'
    )
    width = 1250
    height = 500

    if type_of_food == [] and neighborhood == []:
        folium_static(plot_map(data=data,
                               type_of_food=list(data.type.unique()),
                               neighborhood=list(data.neighborhood.unique()),
                               address=address),
                      width=width,
                      height=height)

    if type_of_food == [] and neighborhood != []:
        folium_static(plot_map(data=data,
                               type_of_food=list(data.type.unique()),
                               neighborhood=neighborhood,
                               address=address),
                      width=width,
                      height=height)

    if type_of_food != [] and neighborhood == []:
        folium_static(plot_map(data=data,
                               type_of_food=type_of_food,
                               neighborhood=list(data.neighborhood.unique()),
                               address=address),
                      width=width,
                      height=height)

    if type_of_food != [] and neighborhood != []:
        folium_static(plot_map(data=data,
                               type_of_food=type_of_food,
                               neighborhood=neighborhood,
                               address=address),
                      width=width,
                      height=height)

    st.markdown(f"#### 💡 Our recommendations 💡 ")

    columns = st.columns(2)

    help_neig = columns[0].button('Help me choose my neighboorhood')
    help_type = columns[1].button('Help me choose my type of restaurant')

    if help_neig:
        st.markdown(
            f'##### Knowing that you want to open a {selected_types} restaurant, here are recommendations for the neighborhoods:'
        )
        st.write(f"**Best neighborhoods** to open a  restaurant(s):")
        #st.write(best_neigh2(data=data, type_of_food=type_of_food))

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "",
            best_neigh(data=data, type_of_food=type_of_food).index[0],
            np.round(best_neigh(data=data,
                                 type_of_food=type_of_food).values[0][0],
                     decimals=1))
        col2.metric(
            "",
            best_neigh(data=data, type_of_food=type_of_food).index[1],
            np.round(best_neigh(data=data,
                                 type_of_food=type_of_food).values[1][0],
                     decimals=1))
        col3.metric(
            "",
            best_neigh(data=data, type_of_food=type_of_food).index[2],
            np.round(best_neigh(data=data,
                                 type_of_food=type_of_food).values[2][0],
                     decimals=1))

        st.write(
            f"**Worse neighborhoods** to open a {selected_types} restaurant:"
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
            f"For {selected_types} restaurants, we recommend a **price range** of **{np.round(best_price_range(data=data, type_of_food=type_of_food),decimals=1)}**"
        )

    if help_type:
        st.markdown(
            f'##### Knowing that you want to open a restaurant in *{selected_neighborhoods}*, here are recommendations for the type of food:'
        )
        #st.markdown(f'##### Help me choose my type of restaurant!')
        st.write(
            f"The **most successful restaurants** in *{selected_neighborhoods}* neighborhood(s) are currently:"
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
            f"The **least successful restaurants** in *{selected_neighborhoods}* neighborhood(s) are currently:"
        )
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
            "The best **price range** in", selected_neighborhoods, 'is',
            np.round(best_price_range_neig(data=data, neighborhood=neighborhood),
                    decimals=1))

        #st.write("The best price range in", neighborhood, 'is',

    if neighborhood ==[]:
        neighborhood = all_neigh(data)

    if type_of_food == []:
        type_of_food = all_types(data)

    if (number_of_restaurant(data=data,
                             type_of_food=type_of_food,
                             neighborhood=neighborhood))<1:
        st.write("Unfortunately, we have no matching restaurant, thus, we cannot provide you with a review analysis")
    else:
        fig = restaurant_full_analysis(df=reviews,
                                    neighborhood=neighborhood,
                                    type_rest=type_of_food)
        st.markdown(
            "#### 📈 Look at the evolution of similar restaurants' ratings and get inspiration for yours! "
        )
        st.write(
            f' Based on the {number_of_restaurant(data=data, type_of_food=type_of_food, neighborhood=neighborhood)} {selected_types} restaurants in {selected_neighborhoods}'
        )

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

    st.markdown(
        f'#### 👉🏼 Gather basic information on your restaurant'
    )

    st.write(
        f'     🍜 Your restaurant is categorized as a *{type_of_food}* restaurant in *{neighborhood}*'
    )

    st.write(
        "      💪 Your current rating is ", rating)


    #Mapping the similar restaurants + the current restaurant in red with a home:
    st.markdown(
        f'#### 📍 Have a look at your restaurant *(in red)* but also at the other *{type_of_food.capitalize()}* restaurants in *{neighborhood}* :'
    )

    #st.write(f'Have a look at your restaurant (in red), but also at the other *{type_of_food.capitalize()}* restaurants in *{neighborhood}* :')
    folium_static(
        plot_map(data,
                 type_of_food=[type_of_food],
                 neighborhood=[neighborhood],
                 latitude2=latitude2,
                 longitude2=longitude2))

    # Plot the evolution of ratings
    #fig = review_evolution(reviews, restaurant=name_of_restaurant)[0]
    #st.pyplot(fig=fig)

    st.markdown("#### 📈 Look at the evolution of your restaurant ratings and understand it ! ")
    fig=restaurant_full_analysis(df=reviews, restaurant=[name_of_restaurant])
    st.pyplot(fig=fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    #if increase >= 0:
    #    st.write("Congrats! On average, ratings of", type_of_food,
    #             "reviews in", neighborhood, 'have increased!')
    #else:
    #    st.write("Unfortunately, look like", type_of_food, "reviews in",
    #             neighborhood, 'have decreased...')

    st.markdown(
        f"#### 🗣️ Evaluate the impact of major sentences of your restaurant' reviews!")

    img = Image.open(f"small_business/data/shapfig/{name_of_restaurant}.png")

    st.image(img)
