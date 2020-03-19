import pandas as pd
raw = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
raw.head(3)
df = raw
df.pop('article_link')
df.dropna()
df.head()
from sklearn.model_selection import train_test_split

X = df['headline']
y = df['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from nltk.corpus import stopwords
import string
import nltk
nltk.download()
from sklearn.feature_extraction.text import CountVectorizer
stop_words =  stopwords.words('english') + list(string.punctuation)
vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words)
X_train = vectorizer.fit_transform(X_train)
from sklearn import naive_bayes

model = naive_bayes.MultinomialNB()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

test_data = vectorizer.transform(X_test)
y_predict = model.predict(test_data)
print(accuracy_score(y_test, y_predict))
sample_data = ['today is sunday',
               'youre tall as a giant dwarf',
               'former versace store clerk sues over secret ',
               'youre very nice little pumpkin !',
              'you are chepal',
              'you are rozeen',
              'i am akshya and i am tall',
              'I work 40 hours a week for us to be this poor',
              'i am shristi and i am sharma',
              'That speaker was so interesting that I barely needed to drink my third cup of coffee.',
              'I am Chepal and im a guitarist',
              'i am akshya and i am a freaking bollywood star',
              'I am little shristi and brand new miss nepal',
              'i promise me and chepal will finish fyp today.. yayy',
              'i am rozeen and i am best programmer in the world yayy yayy yayy',
              'i am rozeen and i am better and richer than bill gates',
              'amazon is my father business',
              'thanks for ruining the day',
              'i am chepal and i am better than rozeen',
              'i am rozeen and better than chepal',
              'Nice perfume. How long did you marinate in it?']
predict_sample_data = vectorizer.transform(sample_data)
predicted = model.predict(predict_sample_data)

for i in range(0, len(sample_data)):
    if predicted[i] == 1:
        print("Sarcastic Statement: ",sample_data[i])
        from test import main
        main(sample_data[i])
    else:
        print(sample_data[i], "-> Non Sarcastic\n")
