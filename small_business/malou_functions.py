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


def get_selection(type_of_food, neighborhood, price, df):
    prediction_row = df.loc[(df['type_of_food'].isin(type_of_food))
                            & (df['neighborhood'].isin(neighborhood)) &
                            (df['price'] == price)]
    return prediction_row
