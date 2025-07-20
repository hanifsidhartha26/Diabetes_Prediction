from sklearn.linear_model import LinearRegression
import numpy as np

sq_ft = np.array([[1500], [1600], [1700], [1800], [1900]])
prices = np.array([300000, 320000, 340000, 360000, 380000])


model = LinearRegression()
model.fit(sq_ft, prices)


new_sq_ft = np.array([[2000]])
predicted_price = model.predict(new_sq_ft)

print(predicted_price[0]    )
