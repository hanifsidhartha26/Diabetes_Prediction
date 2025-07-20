from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
#step1: Data Preparation
data = {
    'area': [1000, 1500, 1800, 2000, 2300, 2500, 2800, 3000],
    'bedrooms': [2, 3, 3, 4, 3, 4, 4, 5],
    'age': [10, 5, 8, 3, 7, 2, 4, 1],
    'price': [30, 45, 50, 60, 65, 70, 85, 90]  # in lakhs
}
df = pd.DataFrame(data)
#step2: Feature Selection
x= df[['area', 'bedrooms', 'age']]
y = df['price']
#step3:train the model
model =LinearRegression()
model.fit(x, y)
#step4:predicting the price of a new house
new_house = pd.DataFrame({'area': [2500], 'bedrooms': [4], 'age': [10]   })
predicted_price = model.predict(new_house)
print(f"The predicted price of the new house is: {predicted_price[0]} lakhs")

