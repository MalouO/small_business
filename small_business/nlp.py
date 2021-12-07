import pandas as pd
import string
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import lime
import sklearn.ensemble
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

class CreateLime():

    def __init__(self, df):
        self.df = df

    def clean(self, text):
        """remove punctuation, lowercases text, removes numbers and returns as string"""
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ') # Remove Punctuation

        lowercased = text.lower() # Lower Case

        tokenized = word_tokenize(lowercased) # Tokenize

        words_only = [word for word in tokenized if word.isalpha()] # Remove numbers

        return " ".join(words_only)

    def logreg_model(self, commentscolumn, ratingscolumn):
        """commentscolumn: please enter the column of the df that contains the comments as a string,
        ratingscolumnn: please enter the column of the df that contains the ratings as a string,
        and """
        #clean and add column for good and bad ratings
        self.df['clean_comment'] = self.df[commentscolumn].apply(self.clean)
        self.df['good_bad_review'] = self.df[ratingscolumn].map(
                lambda x: 1 if x >= 4.0 else 0)

        #split into test and train
        X_train, self.X_test, y_train, y_test = train_test_split(
            self.df['clean_comment'],
            self.df['good_bad_review'],
            random_state=42)

        #run tfidf vectorizer
        self.tf = TfidfVectorizer(strip_accents='ascii', ngram_range=(1, 2))
        X_train_tf = self.tf.fit_transform(
            X_train)  # transform and fit the training set with vectoriser
        X_test_tf = self.tf.transform(self.X_test)  # transform the test set with vectoriser

        #run a logistic regression, fit and use to predict
        logreg = LogisticRegression(verbose=0,
                                    random_state=42,
                                    penalty='l2',
                                    solver='newton-cg',
                                    C=10)
        self.model = logreg.fit(X_train_tf, y_train)
        pred = self.model.predict(X_test_tf)

        metrics.f1_score(y_test, pred, average='weighted')

    def lime_model(self, idx):

        # converting the vectoriser and model into a pipeline, LIME takes a model pipeline as an input
        c = make_pipeline(self.tf, self.model)

        ls_X_test= list(self.X_test)

        # saving the class names in a dictionary to increase interpretability
        class_names = {0: 'bad review', 1:'good review'}

        # create the LIME explainer# add the class names for interpretability
        LIME_explainer = LimeTextExplainer(class_names=class_names)

        idx = idx

        LIME_exp = LIME_explainer.explain_instance(ls_X_test[idx], c.predict_proba)

        # print class names to show what classes the viz refers to
        print("1 = good review, 0 = bad review")
        # show the explainability results with highlighted text
        return LIME_exp.show_in_notebook(text=True)


class CreateNgrams():
    def __init__(self, text):
        self.text = text

    def clean(self):
        """remove punctuation, lowercases text, removes numbers and returns as string"""
        for punctuation in string.punctuation:
            self.text = self.text.replace(punctuation, ' ') # Remove Punctuation

        lowercased = self.text.lower() # Lower Case

        tokenized = word_tokenize(lowercased) # Tokenize

        words_only = [word for word in tokenized if word.isalpha()] # Remove numbers

        return " ".join(words_only)

    def Tfidf_fit_transform(self, series, i):
        """return ngrams of length i for a series"""

        stop_words = set(stopwords.words('english')) # Make stopword list
        #stop_words.update(['food', 'good']) # to be add if we want to remove extra words

        vec = TfidfVectorizer(ngram_range = (i,i), stop_words=stop_words).fit(series)
        vectors = vec.transform(series)# Transform text to vectors

        sum_tfidf = vectors.sum(axis=0) # Sum of tfidf weighting by word

        tfidf_list = [(word, sum_tfidf[0, idx]) for word, idx in     vec.vocabulary_.items()]  # Get the word and associated weight
        sorted_tfidf_list =sorted(tfidf_list, key = lambda x: x[1], reverse=True)  # Sort

        return sorted_tfidf_list[:20]





if __name__ == "__main__":
    df = pd.read_csv("../raw_data/merged_reviews_5_!2.csv")
    lime = CreateLime(df)
    lime.logreg_model('comment_trans','comment_ratings')
    lime.lime_model(12)
