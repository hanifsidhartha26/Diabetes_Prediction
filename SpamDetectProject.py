
# from pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report



# #step 1: Load the dataset

# url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
# df =pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])


# #Step2 : COnvert text to numerical data
# Vectorizer = CountVectorizer()
# x= Vectorizer.fit_transform(df['message'])
# y=df['label']

# #Step 3: Split the dataset into training and testing sets

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# #Step 4: Train the model using Naive Bayes classifier

# model = MultinomialNB()
# model.fit(x_train, y_train)

# #Step 5: Make predictions on the test set
# y_pred = model.predict(x_test)


# #Step 6: Evaluate the model's performance
# print("Accuracy:", accuracy_score(y_test, y_pred))





# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Step 2: Load sample spam dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# Step 3: Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5: Train classifier
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 8: Test on new message
sample = vectorizer.transform(["Congratulations! You've won a free ticket!"])
print("Prediction:", model.predict(sample)[0])
