from save_model import *

X, y = restaurant_data()
pipeline = get_fitted_pipe(X, y)
neighborhood = neighborhood
address = address
latitude, longitude = address_imputer(address, neighborhood)
X_user = target_X(type_of_food = type_of_food, price = price, neighborhood = neighborhood, takeaway = 1, latitude = latitude, longitude=longitude)
X_user_transformed = pipeline.transform(X_user)
y_class = build_y(y)
predict(X_user_transformed)
neighbours(X_user_transformed)
