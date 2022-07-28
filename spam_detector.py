import pickle

import nltk
import pandas as pd
from importlib_metadata import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from globall import MLModel
from lgbm_classifier import LGBMClassifier
from supplier import Supplier


class SpamDetector:
    LABEL_TITLE = 'label'
    MESSAGE_TITLE = 'message'

    def __init__(self, dataset_path):
        # check if dataset's path is empty
        if not dataset_path or len(dataset_path) == 0:
            raise ValueError('Dataset path should be defined')

        # check if dataset's file does not exist
        if not os.path.exists(dataset_path):
            raise FileNotFoundError('There is no dataset file in the give address')

        self._dataset_path = dataset_path
        # fill the dataframe with dataset contents
        self._df = pd.read_csv(dataset_path, encoding='latin-1')
        self._data_ham = None
        self._data_spam = None
        self._tfidf_data = None

        # download stopwords for nltk
        nltk.download('stopwords')

    def get_df(self):
        return self._df.copy()

    def clean_dataset(self):
        self._df.drop(columns=self._df.iloc[:, 2:], axis=1, inplace=True)
        self._df.rename(columns={'v1': self.LABEL_TITLE, 'v2': self.MESSAGE_TITLE}, inplace=True)

    def feature_engineering(self):
        EX_FEATURE_1 = 'length'
        self._df[EX_FEATURE_1] = self._df[self.MESSAGE_TITLE].apply(len)
        return EX_FEATURE_1

    def ham_spam_splitting(self):
        self._data_ham = self._df[self._df[self.LABEL_TITLE] == 'ham'].copy()
        self._data_spam = self._df[self._df[self.LABEL_TITLE] == 'spam'].copy()

    def get_data_ham(self):
        if self._data_ham is None:
            raise ValueError('First, ham_spam_splitting should be run')
        return self._data_ham

    def get_data_spam(self):
        if self._data_spam is None:
            raise ValueError('First, ham_spam_splitting should be run')
        return self._data_spam

    def clean_text_messages(self):
        # Replace string of labels with numbers
        self._df[self.LABEL_TITLE] = self._df[self.LABEL_TITLE].map({'spam': 1, 'ham': 0})

        # Replace email address with '##emailaddress##'
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                                                                '##emailaddress##')

        # Replace urls with '##webaddress##'
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].str.replace(
            r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', '##webaddress##')

        # Replace money symbol with '##moneysymbol##'
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].str.replace(r'£|\$', '##moneysymbol##')

        # Replace 10 digit phone number with '##phonenumber##'
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].str.replace(
            r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', '##phonenumber##')

        # Replace normal number with '##number##'
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].str.replace(r'\d+(\.\d+)?', '##number##')

        # Remove punctuations
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].str.replace(r'[^\w\d\s]', ' ')

        # Remove whitespaces between terms with single space
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].str.replace(r'\s+', ' ')

        # Remove leading and trailing whitespaces
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].str.replace(r'^\s+|\s*?$', ' ')

        # Change words to lowercase
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].str.lower()

    def remove_stop_words(self):
        # load stopwords for english
        stop_words = set(stopwords.words('english'))
        # remove stopwords
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].apply(
            lambda x: ' '.join(term for term in x.split() if term not in stop_words))

    def stemming_words(self):
        # load the model for stemming english words
        ss = nltk.SnowballStemmer('english')
        # stemming words
        self._df[self.MESSAGE_TITLE] = self._df[self.MESSAGE_TITLE].apply(
            lambda x: ' '.join(ss.stem(term) for term in x.split()))

    def create_bag_of_words(self):
        # Vectorzing words
        self._msg_df = self._df[self.MESSAGE_TITLE]
        # creating a bag-of-words model
        all_words = []
        for sms in self._msg_df:
            words = word_tokenize(sms)
            for w in words:
                all_words.append(w)

        all_words = nltk.FreqDist(all_words)
        print('Number of words: %d' % (len(all_words)))
        all_words.plot(10, title='Top 10 Most Common Words in Corpus')

    def create_tfidf_model(self):
        tfidf_model = TfidfVectorizer()
        tfidf_vec = tfidf_model.fit_transform(self._msg_df)

        # Serializing the model to a file called model.pkl
        pickle.dump(tfidf_model, open('model/tfidf_model.pkl', 'wb'))
        self._tfidf_data = pd.DataFrame(tfidf_vec.toarray())

    def build_model(self, model: MLModel):
        supplier = Supplier(LGBMClassifier.MODEL_NAME, self._tfidf_data, self.get_df()[self.LABEL_TITLE])
        if model == MLModel.GBM:
            classifier = LGBMClassifier(supplier)
        else:
            raise ValueError('Classifier model name should be defined')
        model = classifier.fit()
        supplier.dump_full_model(model)
