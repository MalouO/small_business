import pandas as pd
import string
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
import lime
import sklearn.ensemble
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

def clean(text):
    """remove punctuation, lowercases text, removes numbers and returns as string"""
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation

    lowercased = text.lower() # Lower Case

    tokenized = word_tokenize(lowercased) # Tokenize

    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers

    return " ".join(words_only)

def logreg_model(df):
    df['clean_comment'] = df['comment_trans'].apply(clean)
    df['good_bad_review'] = df.comment_ratings.map(
            lambda x: 1 if x >= 4.0 else 0)

    cleaned_merged_df = df.sample(frac=1)

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_merged_df['clean_comment'],
        cleaned_merged_df['good_bad_review'],
        random_state=42)

    tf = TfidfVectorizer(strip_accents='ascii', ngram_range=(1, 2))

    X_train_tf = tf.fit_transform(
        X_train)  # transform and fit the training set with vectoriser
    X_test_tf = tf.transform(X_test)  # transform the test set with vectoriser

    logreg = LogisticRegression(verbose=0,
                                random_state=42,
                                penalty='l2',
                                solver='newton-cg',
                                C=10)

    model = logreg.fit(X_train_tf, y_train)

    pred = model.predict(X_test_tf)
    metrics.f1_score(y_test, pred, average='weighted')

def lime_model():
    pass
