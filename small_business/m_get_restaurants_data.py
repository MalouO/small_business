import pandas as pd

def restaurant_data():

    data = pd.read_csv('../small_business/data/restaurants.csv')
    data = data.drop(columns = 'Unnamed: 0')
    X = data.drop(columns=['rating','name', 'address', 'label', 'curb_pickup', 'drive_through', 'postal_code', 'no_del_exp', 'municipality', 'review_count'])
    y = data['rating']
    return X, y
