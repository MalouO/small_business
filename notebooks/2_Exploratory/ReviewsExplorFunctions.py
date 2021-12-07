import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('english')) # Make stopword list


# Tuned TFidfvectorizer
def Tfidf_fit(series):
    vec = TfidfVectorizer(ngram_range = (2,2), stop_words=stop_words).fit(series)
    return vec



def transform_create_list(series):
    vectors = Tfidf_fit(series).transform(series) # Transform text to vectors

    sum_tfidf = vectors.sum(axis=0) # Sum of tfidf weighting by word

    tfidf_list = [(word, sum_tfidf[0, idx]) for word, idx in Tfidf_fit(series).vocabulary_.items()]  # Get the word and associated weight

    sorted_tfidf_list =sorted(tfidf_list, key = lambda x: x[1], reverse=True)  # Sort

    return sorted_tfidf_list[:7]



def restaurant_list(df, restaurant=None, neighborhood=None, type_rest=None, price=None):

    if restaurant != None:
        df = df[df['restaurant_name'].isin(['restaurant'])]
        return restaurant

    if neighborhood != None:
        df = df[df['neighborhood'].isin(neighborhood)]
    
    if type_rest != None:
        df = df[df['type'].isin(type_rest)]
    
    if price != None:
        df = df[df['price'].isin(price)]

    if df.shape[0] == 0:
        return "No restaurants found."
        
    else:
        restaurant = list(df['restaurant_name'].unique())
        
    return restaurant



def review_barplot(df, restaurant, title):

    df = df[df['restaurant_name'].isin(restaurant)]
    data = df.groupby('monthyear').agg({'comment_dates':'mean','comment_ratings':'mean','review_count':'count', 'comment_trans':transform_create_list})
    data = data.rename(columns={'comment_dates':'Dates','comment_ratings':'Ratings','review_count':'#Comments','comment_trans':'Tfidf'})
    data['Tfidf'] = [i for i in data['Tfidf']]
    plt.figure(figsize=(20,10))
    data.sort_values(by='Dates', inplace=True)
    f=sns.barplot(data=data, x='Dates', y='#Comments', color='#ec645c', edgecolor='black')
    
    for k, v in enumerate(data['Tfidf']):
        f.text(k, data['#Comments'].values[k], ',\n'.join([i[0] for i in v[:3]]), ha='center', color='black', size=9, bbox={'facecolor': '#f58eaf', 'alpha': 1, 'pad': 10})
        if data['#Comments'].values[k] > max(data['#Comments'].values)*0.1:
            f.text(k, 1, data['#Comments'].values[k], va='bottom', ha='center', color='black', size=14, weight='bold', fontname="Arial")

    f2 = f.twinx()
    f2.plot(data['Ratings'].values, linestyle='-', linewidth=2, color='#f4c46c', marker='o', markerfacecolor='#f4c46c', markeredgecolor='black', markersize=30, markeredgewidth=0.5, label='Rating')
    for k, v in enumerate(data['Ratings']):
        f2.text(k, data['Ratings'].values[k]-0.04, round(v,1), color='black', ha='center', size=12, weight='bold', fontname="Arial")
    
    
    f.spines["right"].set_visible(False)
    f.spines["left"].set_visible(False)
    f.spines["top"].set_visible(False)

    f2.spines["left"].set_visible(False)
    f2.spines["top"].set_visible(False)
    f2.spines["right"].set_visible(False)

    
    f2.set_ylim(0,6)
    
    f.set_yticks([])
    f2.set_yticks([])
    
    labels = [i.year for i in data.Dates]
    f.set_xticklabels(labels, size=15, fontname="Arial")
    
    f.set_xlabel('')
    f.set_ylabel('')
    
    plt.title(f'{title}    ({len(restaurant)} restaurants   -   {df.shape[0]} reviews)', loc='left', size=15, fontname="Arial")
    
    return f

def restaurant_full_analysis(df, restaurant=None, neighborhood=None, type_rest=None, price=None):
    
    df = df.copy()
    df['comment_dates'] = pd.to_datetime(df['comment_dates'])
    df['monthyear'] = df['comment_dates'].map(lambda x : int(f'{x.year}'))

    if restaurant != None:
        title = restaurant
    else:
        title = (f'Neighborhoods : {neighborhood}   -   Categories : {type_rest}   -   Price ranges : {price}')

    restaurant = restaurant_list(df, restaurant, neighborhood, type_rest, price)

    #review_scatter(df, restaurant, title)
    review_barplot(df, restaurant, title)



## CALL THE FUNCTION
#reviews = pd.read_csv('../../small_business/data/reviews.csv')

#restaurant = None
#neighborhood = None
#type_rest = None
#price = None

# either this
#restaurant = ['Flor da Laranja']

# or this
#neighborhood = ['Encarnação']
#type_rest = ['brunch']
#price = [1]

#restaurant_full_analysis(reviews, restaurant=restaurant, neighborhood=neighborhood, type_rest=type_rest, price=price)