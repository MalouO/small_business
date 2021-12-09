from prediction import *

df = pd.read_csv('../notebooks/3_Prediction/data_probabilities.csv')
#df = df.drop(columns='Unnamed : 0')

get_selection(type_of_food=['pizza', 'brunch'], neighborhood=['Lapa', 'Graça'], price=2, df=df)
eval_idea(
    get_selection(type_of_food=['pizza', 'brunch'],
                  neighborhood=['Lapa', 'Graça'],
                  price=2,
                  df=df))
