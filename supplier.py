import pickle

from sklearn import model_selection
from sklearn.metrics import f1_score


class Supplier:
    def __init__(self, model_name, data, target):
        self._model_name = model_name
        self._data = data
        self._target = target

    def train_test_splitting(self):
        # Train-Test Splitting
        train_size = int(len(self._data) * 0.7)
        df_train = self._data.iloc[:train_size]
        df_test = self._data.iloc[train_size:]
        df_train['label'] = self._target

        Y = df_train['label']
        X = df_train.drop('label', axis=1)

        # Splitting training data into train and validation using sklearn
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

    def get_X_train(self):
        return self.X_train

    def get_y_train(self):
        return self.y_train

    def predict(self, model):
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        return pred

    def dump_full_model(self, model):
        model.fit(self._data, self._target)
        pickle.dump(model, open('model/spam_model.pkl', 'wb'))
        
    def get_f1_score(self, pred):
        return f1_score(pred, self.y_test)
