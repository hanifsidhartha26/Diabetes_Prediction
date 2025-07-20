#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#load data 

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url,names=cols)
print(df.head())


#data Analysis
print(df.describe())
print(df.info())
print(df.isnull().sum())

#visualize correlations
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature correlation")
plt.show()

#outcome distribution
sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome Count (0=No, 1=Yes)")
plt.show()


#train spilit & model training
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#split data
x=df.drop('Outcome', axis=1)
y=df['Outcome']

#normalize data
scaler = StandardScaler()
x_sclaed =scaler.fit_transform(x)

#train test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

#train model
model = LogisticRegression()
model.fit(x_train,y_train)

#predict
y_predict = model.predict(x_test)


#evaluate
print("Accuracy :" , accuracy_score(y_predict,y_test))
print("confusion matrix :", confusion_matrix(y_test,y_predict) )
print("Classification Report:", classification_report(y_test, y_predict))


# Predict on a custom input
sample = [[6,148,72,35,0,33.6,0.627,50]]  # sample input
sample_scaled = scaler.transform(sample)
result = model.predict(sample_scaled)

print("Prediction (0 = No diabetes, 1 = Diabetes):", result[0])
