from sklearn.linear_model import LinearRegression
import numpy as np

exp = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
sal = np.array([1000,1500,2000,2500,3000])

model = LinearRegression()
model.fit(exp,sal)


print("next salary for 6 years of experience is: ", model.predict([[6]]))