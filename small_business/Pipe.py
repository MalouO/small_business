import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('../raw_data/Clean_data_1_12_v2.csv')
def deletespace(x):
    return x.strip()

def replace_type(x):
    result=x
    for k, v in dico.items():
        if x in v:
            result=k
    return result


dico={}
dico['restaurant']=['restaurant', 'restaurante', 'family']
dico['cafe']=['cafe', 'tea room', 'bubble tea store', 'torrefatores de café','coffee shop', 'coffee roasters', 'art cafe', 'café', 'espresso bar', 'coffee store']
dico['italian']=['italian']
dico['pizza']= ['pizza takeaway', 'pizza', 'pizaria']
dico['fast_food']=['fast food', 'hamburger', 'comida rápida', 'sandwich shop',  'kebab shop', 'hot dog stand', 'hot dog ', 'fried chicken takeaway', 'chicken']
dico['tapas']=['tapas', 'tapas bar']
dico['bar']=['pub', 'bar', 'cocktail bar', 'gastropub']
dico['brunch']=['brunch',  'breakfast', 'restaurante de brunch']
dico['show']=['fado', 'dinner theatre']
dico['bakery_pastry']=['pastry shop','dessert shop', 'bakery' ]
dico['grill']=['grill', 'barbecue','steak house' ]
dico['veggie_healthy']= ['health food', 'vegan', 'vegetarian']
dico['japanese']=['sushi', 'japanese']
dico['american']=['american', 'diner']
dico['african']=[ 'african', 'moroccan']
dico['asian']=['asian', 'pan-asian', 'vietnamese',  'thai',  'chinese']
dico['european']=['european', 'modern european']
dico['indian']=['indian muslim', 'indian', 'bangladeshi', 'restaurante nepalês', 'nepalese ']
dico['mexican']=['restaurante mexicano','mexican']
dico['south_am']=['peruvian', 'argentinian']
dico['middle_eastern']=['middle eastern', 'turkish','georgian', 'halal',  'restaurante halal' ]
dico['bistro']=['bistro']
dico['traditional']=['traditional']
dico['international_rest']=['belgian', 'austrian', 'australian', 'french']
dico['mediterranean']=['mediterranean', 'andalusian', 'basque']
dico['fine-dining']=['fine-dining']
dico['seafood']=['seafood']
dico['portuguese']=['portuguese']


def pipeline(data, test_size = 0.2):
    data['type_gen']=data.type.map(deletespace)
    data['type_gen']=data.type.map(replace_type)
    data=data[data.type_gen != 'out1']
    #data = data[data.type_gen != 'restaurant']
    X = data[['review_count', 'price', 'dine_in', 'takeaway', 'delivery','drive_through', 'type_gen','no_del_exp', 'curb_pickup', 'neighborhood']]
    y = data['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    imputer = SimpleImputer(strategy="most_frequent")
    imputer.fit(X_train[['price']])
    X_train['price']= imputer.transform(X_train[['price']])
    X_test['price']= imputer.transform(X_test[['price']])
    ohe = OneHotEncoder(handle_unknown='ignore', sparse = False)
    ohe.fit(X_train[['price', 'neighborhood', 'type_gen']])
    X_train_encoded = pd.DataFrame(ohe.transform(X_train[['price', 'neighborhood', 'type_gen']]))
    X_test_encoded = pd.DataFrame(ohe.transform(X_test[['price', 'neighborhood', 'type_gen']]))
    X_test_encoded.columns = ohe.get_feature_names_out()
    X_train_encoded.columns = ohe.get_feature_names_out()
    X_train= X_train.join(X_train_encoded, how='left')
    X_test = X_test.join(X_test_encoded, how='left')
    X_train = X_train.drop(columns = ['neighborhood', 'type_gen', 'review_count'])
    X_test = X_test.drop(columns = ['neighborhood', 'type_gen', 'review_count'])
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    return  X_train, X_test, y_train, y_test
