import os
import sys

from dataclasses import dataclass

from catboost import CatBoostClassifier


from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models


@dataclass

class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.mode_trainer_config=ModelTrainerConfig()

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('spliting train and test input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params = {
            "Decision Tree": {
                # Classification uses gini or entropy, NOT squared_error
                'criterion': ['gini', 'entropy', 'log_loss'],
                'splitter': ['best', 'random'],},
            "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
            "Gradient Boosting": {
                # Classification loss is 'log_loss' or 'exponential'
                'loss': ['log_loss'], 
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            # Replace Linear Regression with Logistic Regression
            "Logistic Regression": {}, 
            # Replace XGBRegressor with XGBClassifier
            "XGBClassifier": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            # Replace CatBoosting Regressor with CatBoostClassifier
            "CatBoost Classifier": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            # Replace AdaBoost Regressor with AdaBoostClassifier
            "AdaBoost Classifier": {
                'learning_rate': [0.1, 0.01, 0.5, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
}

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,
                                             param=params
                                             )
            
            #get the best model

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info(f"best model found on both training and testing datasets")

            save_object(
                file_path=self.mode_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy=accuracy_score(y_test,predicted)

            return accuracy








        except Exception as e:
            raise CustomException(e,sys)
            





