import pandas as pd
def train():
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
    return model,vectorizer

def test(statement,model,vectorizer):
    statementList = [statement]
    import string
    predict_sample_data = vectorizer.transform(statementList)
    predicted = model.predict(predict_sample_data)
    if predicted== 1:
        print("Sarcastic Statement: ",statement)
        from test import main
        ret_text=main(statement)
        return ret_text
    else:
            print(statement, "-> Non Sarcastic\n")
            ret_text=''
            return ret_text
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
result=train()
for i in range(len(sample_data)):
    #print('Statement: ',sample_data[i])
    ret_val=test(sample_data[i],result[0],result[1])
    print('Response: ',ret_val)#empty response indicates input statement wasn't deemed sarcastic

