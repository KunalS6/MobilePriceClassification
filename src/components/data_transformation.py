import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging


import os

from src.utils import save_object




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transforamtion=DataTransformationConfig()
    

    def get_transformer_object(self):
        """
        this function deals with data transformation
        
        """



        try:
            numerical_features=['battery_power',
                                'clock_speed',
                                'fc','int_memory',
                                'm_dep',
                                'mobile_wt',
                                'n_cores',
                                'pc',
                                'px_height',
                                'px_width',
                                'ram',
                                'sc_h',
                                'sc_w',
                                'talk_time']
            categorical_features=['blue', 
                                  'dual_sim', 
                                  'four_g', 
                                  'three_g', 
                                  'touch_screen', 
                                  'wifi']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Numerical columns {numerical_features}')

            logging.info(f'Categorical columns {categorical_features}')

            preprocessor=ColumnTransformer(
                [                
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_features)
                    ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_transformer_object()

            target_column_name='price_range'
            numerical_features=['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width','ram','sc_h','sc_w','talk_time']

            # train dataset

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            # test dataset

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing object')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"saved preprocessing object.")

            save_object(
                file_path=self.data_transforamtion.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transforamtion.preprocessor_obj_file_path)
        








        except Exception as e:
            raise CustomException(e,sys)
            