from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

emails = np.array([
    "Congratulations! You've won a lottery.",
    "Dear friend, I need your help with a business proposal.",  
    "Get rich quick with this simple trick!",
    "Important: Update your account information immediately.",
    "Hello, how are you doing today?",
    "This is not spam, just a friendly reminder."
    ])

labels = np.array([1, 0, 1, 1, 0, 0])  # 1 for spam, 0 for not spam

#convert text into numerical data
vectorizer = CountVectorizer()
x =vectorizer.fit_transform(emails)

#train the model
model = MultinomialNB()
model.fit(x, labels)

#predict new mail text

new_email = ["Congratulations! You have won a free vacation!"]
new_email_vectorized = vectorizer.transform(new_email)

print("Is the new email spam? ", model.predict(new_email_vectorized)[0])