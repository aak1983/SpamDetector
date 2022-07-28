import lightgbm as lgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from supplier import Supplier


class LGBMClassifier:
    PARAM_GRID = {
    'num_leaves': list(range(8, 92, 4)),
    'min_data_in_leaf': [10, 20, 40, 60, 100],
    'max_depth': [3, 4, 5, 6, 8, 12, 16, -1],
    'learning_rate': [0.1, 0.05, 0.01, 0.005],
    'bagging_freq': [3, 4, 5, 6, 7],
    'bagging_fraction': np.linspace(0.6, 0.95, 10),
    'reg_alpha': np.linspace(0.1, 0.95, 10),
    'reg_lambda': np.linspace(0.1, 0.95, 10),
    'min_split_gain': [0.0, 0.1, 0.01],
    'min_child_weight': [0.001, 0.01, 0.1, 0.001],
    'min_child_samples': [20, 30, 25],
    'subsample': [1.0, 0.5, 0.8],
}
    MODEL_NAME = 'Light GBM'
    default_n_estimators = 200
    default_num_leaves = 40

    def __init__(self, supplier:Supplier):
        self._supplier = supplier
        supplier.train_test_splitting()

    def _find_best_depth(self):
        best_depth = 1
        best_score = 0
        for depth in range(1, 11):
            lgbmodel = lgb.LGBMClassifier(max_depth=depth, n_estimators=self.default_n_estimators, num_leaves=self.default_num_leaves)
            print(f'Max Depth {depth}')
            pred = self._supplier.predict(lgbmodel)
            f1_score = self._supplier.get_f1_score(pred)
            print('F1 Score is: %.6f\n' % (f1_score))
            if f1_score > best_score:
                best_depth = depth

        return best_depth

    def _search_best_params(self):
        best_depth = self._find_best_depth()
        lgbmodel_best = lgb.LGBMClassifier(max_depth=best_depth, n_estimators=self.default_n_estimators, num_leaves=self.default_num_leaves)
        model = RandomizedSearchCV(lgbmodel_best, self.PARAM_GRID, random_state=1)
        search = model.fit(self._supplier.get_X_train(), self._supplier.get_y_train())
        return search.best_params_

    def _fit_by_best_model(self, best_params):
        best_model = lgb.LGBMClassifier(**best_params)
        pred = self._supplier.predict(best_model)
        print('F1 Score is: %.5f' % (self._supplier.get_f1_score(pred)))
        return best_model

    def fit(self):
        best_params = self._search_best_params()
        return self._fit_by_best_model(best_params)
