import string

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

from nltk import WordNetLemmatizer
from nltk.corpus import wordnet


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def text_data_preprocessing(x):
    tknzr = TweetTokenizer()
    tokenized_text = pd.Series(tknzr.tokenize(i.lower()) for i in x)

    noise = stopwords.words('english')
    t_t_without_noise = pd.Series([x for x in i if x not in noise] for i in tokenized_text)

    lemmatizer = WordNetLemmatizer()
    lemmatized_text = pd.Series([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in i] for i in t_t_without_noise)

    processed_text = pd.Series(" ".join(i) for i in lemmatized_text)

    for i in range(len(processed_text)):
        for ch in string.punctuation:
            processed_text[i] = processed_text[i].replace(ch, "")

    return processed_text


data = pd.read_csv('/Users/dsivan/Downloads/sms_spam.csv')
x_train, x_test, y_train, y_test = train_test_split(data.text, data.type, train_size=0.7)

x_train_processed = text_data_preprocessing(x_train)

x_test_processed = text_data_preprocessing(x_test)

# CountVectorizer. Words

pipeline = Pipeline([
           ('vect', CountVectorizer()),
           ('clf', MultinomialNB()),
])

params = {
    'vect__ngram_range': [(k, i) for i in range(2, 6) for k in range(1, i + 1)]
}

search = GridSearchCV(pipeline, params, scoring='f1_macro')
search.fit(x_train_processed, y_train)
print(search.best_score_)
print(search.best_params_)

pred = search.predict(x_test_processed)
print(classification_report(y_test, pred))

# CountVectorizer. Chars

pipeline2 = Pipeline([
           ('vect', CountVectorizer(analyzer='char')),
           ('clf', MultinomialNB()),
])

params2 = {
    'vect__ngram_range': [(k, i) for i in range(2, 8) for k in range(1, i + 1)]
}

search2 = GridSearchCV(pipeline2, params2, scoring='f1_macro')
search2.fit(x_train_processed, y_train)
print(search2.best_score_)
print(search2.best_params_)

pred2 = search2.predict(x_test_processed)
print(classification_report(y_test, pred2))

# TfidfVectorizer

pipeline3 = Pipeline([
           ('vect', TfidfVectorizer()),
           ('clf', MultinomialNB()),
])

params3 = {
    'vect__ngram_range': [(k, i) for i in range(2, 6) for k in range(1, i + 1)],
    'vect__min_df': [4, 5, 2],
    'vect__max_df': [0.75, 0.85, 0.65],
    'vect__max_features': [4000, 5000, 6000]
}


search3 = GridSearchCV(pipeline3, params3, scoring='f1_macro', n_jobs=-1)
search3.fit(x_train_processed, y_train)
print(search3.best_score_)
print(search3.best_params_)

pred3 = search3.predict(x_test_processed)
print(classification_report(y_test, pred3))
vect = dict()