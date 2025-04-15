import pandas as pd
import re
import demoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import string
from sklearn.neighbors import KNeighborsClassifier
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import OneHotEncoder

# Load CSV data into a DataFrame
df = pd.read_csv('sentimentdataset.csv')

# Select only the columns you want to keep
columns_to_keep = ['Text', 'Sentiment (Label)','Topic']
df = df[columns_to_keep]


# Preprocessing

sia = SentimentIntensityAnalyzer()
def classify_sentiment(word):
    score = sia.polarity_scores(word)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df['Sentiment (Label)'] = df['Sentiment (Label)'].apply(classify_sentiment)


#Encoding
encoder = LabelEncoder()
df["Sentiment (Label)"] = encoder.fit_transform(df["Sentiment (Label)"])

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = demoji.replace(text, "")  # Remove emojis
    return text


df['Text'] = df['Text'].apply(preprocess_text)

# Tokenization
df['Text'] = df['Text'].apply(word_tokenize)

# Remove stopwords
stopwords_En = set(stopwords.words('english'))
df['Text'] = df['Text'].apply(lambda x: [word for word in x if word not in stopwords_En])

# Lemmatization
lemmatizer = WordNetLemmatizer()
df['Text'] = df['Text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Convert tokens back to string
df['Text'] = df['Text'].apply(lambda x: ' '.join(x))

# Oversample the minority class
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(df['Text'].values.reshape(-1, 1), df['Sentiment (Label)'])

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=1)


# Vectorization
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train.ravel())
X_test_vectorized = vectorizer.transform(X_test.ravel())


# # Define the RNN model
# model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=(X_train_vectorized.shape[1],)))
# model.add(Dense(3, activation='softmax'))
#
# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train_vectorized, y_train, epochs=10, batch_size=64, validation_split=0.1)
#
# # Model evaluation on training set
# train_loss, train_accuracy = model.evaluate(X_train_vectorized, y_train)
# print("Training Accuracy:", train_accuracy)
#
# # Model evaluation on test set
# test_loss, test_accuracy = model.evaluate(X_test_vectorized, y_test)
# print("Test Accuracy:", test_accuracy)



#Models

# Logistic Regression Model
model = LogisticRegression()

#KNN Model
#model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k)

#Naive Model
#model= MultinomialNB()

#SVC Model
#model= SVC(kernel='linear')




# Fit the model to the training data
model.fit(X_train_vectorized, y_train)

# Now you can make predictions
y_train_pred = model.predict(X_train_vectorized)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

# Make predictions on the test set
y_test_pred = model.predict(X_test_vectorized)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)
