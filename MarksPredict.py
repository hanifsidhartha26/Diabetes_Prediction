import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#load data set 
url ="http://bit.ly/w-data"
df = pd.read_csv(url)
print(df.head())


#visualise data

plt.scatter(df['Hours'], df['Scores'] , color = 'green')
plt.title("Hours vs Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()



#prepare for training

x = df[["Hours"]]
y = df["Scores"]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)


#train the model
model = LinearRegression()
model.fit(x_train, y_train)

#prediction

y_pred = model.predict(x_test)

#compare
df = pd.DataFrame({"Actual": y_test,'predict' : y_pred} )
print(df)


#mse
mse = mean_squared_error(y_test,y_pred)
print(mse)

#sample
hours = [[7.5]]  # Enter hours studied
predicted_score = model.predict(hours)
print("Predicted Score for 7.5 hours:", predicted_score[0])
