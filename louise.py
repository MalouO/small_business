from small_business.Victor_functions import *

name_of_restaurant='Dear Breakfast'
reviews = pd.read_csv('raw_data/reviews.csv')

restaurant_full_analysis(df=reviews, restaurant=[name_of_restaurant])
plt.show()
