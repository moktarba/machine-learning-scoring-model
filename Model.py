
# 1. Library imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pydantic import BaseModel

class CreditModel(BaseModel):
    index : int
    #random_forest = RandomForestClassifier()
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and
    #    saves the model
    def __init__(self):
        self.df = pd.read_csv('data.csv')
        self.model_fname_ = 'random_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            #self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)

    def predict(data) :
        prediction = self.model.predict_proba(data)
        return prediction[0]

    def _train_model(self):
        X = self.df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        y = self.df.TARGET
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        random_best = RandomForestClassifier(class_weight='balanced', max_depth=5,min_samples_leaf=32)
        # Training the basic Decision Tree model with training set
        model = random_best.fit(X_train,y_train)
        return model
