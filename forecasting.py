import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import os
import gc
import random
from argparse import ArgumentParser
from tqdm import tqdm
from utils.preprocessing import preprocessing
from utils.evaluation import evaluation
from models import *

tf.config.list_physical_devices('GPU')


class Forecasting:
    def __init__(self, args):
        
        self.code = args.code
        self.type = args.type
        self.lookback = args.lookback
        self.scaler = args.scaler
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.callbacks = args.callbacks
        self.model = args.model
        self.column_target = 'Close'
        
        self.models_graph = ['TFGCNGRU','TFGCNLSTM']
        self.is_graph = True if self.model in self.models_graph else False
        
        if args.list_of_code:
            self.IssuerCode(True)
        else:
            path_result = 'results/'
            self.path_result_evaluations = path_result+'evaluations/'
            self.path_result_models = path_result+'models/'
            self.path_result_plots = path_result+'plots/'
            self.path_result_datas = path_result+'targets/'

            self.create_folder(path_result)
            self.create_folder(self.path_result_evaluations)
            self.create_folder(self.path_result_models)
            self.create_folder(self.path_result_plots)
            self.create_folder(self.path_result_datas)            

            self.ProcessForecasting()

    def create_folder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
            
    def IssuerCode(self, state=False):
        issuers = pd.read_csv('data/s&p500.csv',sep=';')
        issuer_filtered = []
        for code_stock in tqdm(issuers.Ticker, desc='Filtering Issuer'):
            # Check last transaction 2023
            # Have 200 transaction
            path = f'data/transactions/{code_stock}.csv'
            if os.path.exists(path):
                historical = pd.read_csv(path)
                last_date_row = historical.Date.iloc[len(historical)-1] # get last date transactions
                data_test = historical[len(historical)-(int(len(historical)*0.2)):][self.column_target] # get data test
                var = np.var(data_test) # calculate variance from data test
                if (last_date_row[:4] == '2023' and len(historical) >= 200 and var > 0):
                    issuer_filtered.append(code_stock)
        
        if state:
            print("Code of Stock Issuer: {}".format(issuer_filtered))
        else:
            return issuer_filtered

    def ProcessForecasting(self):
        issuer = self.IssuerCode()
        if self.code == 'All':
            codes = issuer
        else:
            codes = [item.strip() for item in self.code.split(',')]
        lookbacks = [int(item.strip()) for item in self.lookback.split(',')]

        for code in codes:
            if code not in issuer:
                code_stock = code
                print(f'\n Code {code_stock} is not valid, please check command')
            else:
                print('\n############# Load dataset...')
                code_stock = code
                path = f'data/transactions/{code_stock}.csv'
                if os.path.exists(path):
                    historical = pd.read_csv(path)
                historical.Date = pd.to_datetime(historical.Date).dt.date
                historical = historical.set_index(historical.Date)
                historical = historical.drop(['Date'], axis=1)
                
                
                for lookback in lookbacks:
                    fix_seed = 2023
                    random.seed(fix_seed)
                    tf.random.set_seed(fix_seed)
                    np.random.seed(fix_seed)
                    
                    dataset = historical
                    prediction_days = int(len(dataset) * 0.2)

                    print('\n############# Preprocessing split data train and test...')
                    if self.type == 0:
                        type_dataset = 'univariate'
                        pp = preprocessing(
                            self.scaler, f'{self.model}_{type_dataset}', f'{code_stock}_{lookback}', self.path_result_datas
                        )
                        x_train, x_test, y_train, y_test = pp.dataset_univariate(
                            dataset[self.column_target], lookback, prediction_days, self.batch_size, )
                    elif self.type == 1:
                        type_dataset = 'multivariate'
                        pp = preprocessing(
                            self.scaler, f'{self.model}_{type_dataset}', f'{code_stock}_{lookback}', self.path_result_datas
                        )
                        x_train, x_test, y_train, y_test = pp.dataset_multivariate(
                            dataset, self.column_target, lookback, prediction_days, self.batch_size)
                    elif self.type == 2:                        
                        type_dataset = 'multivariate_graph'
                        pp = preprocessing(
                            self.scaler, f'{self.model}_{type_dataset}', f'{code_stock}_{lookback}', self.path_result_datas
                        )
                        x_train, x_test, y_train, y_test, a_train, a_test = pp.dataset_graph(
                            dataset, self.column_target, lookback, prediction_days, self.batch_size)

                    self.create_folder(
                        f'{self.path_result_models}{self.model}_{type_dataset}/')
                    
                    checkfiles = f'{self.path_result_evaluations}{self.model}_{type_dataset}/{code_stock}_{lookback}_test_score.json'
                    if os.path.exists(checkfiles) == False:
                        print(f'\n############# Build model {self.model} {type_dataset} {code_stock}...')
                        module_globals = globals()
                        module = module_globals[self.model]
                        architecture = module(n_classes=1)
                        # Graph
                        if(self.is_graph):
                            model = architecture.build_model(x_input_shape=x_train.shape[1:], g_input_shape=a_train.shape[1:])
                        else:
                            model = architecture.build_model(input_shape=x_train.shape[1:])
                        
                        model.compile(
                            loss="mse",
                            optimizer=tf.keras.optimizers.Adam(
                                learning_rate=1e-3, decay=1e-3),
                            metrics=[tf.keras.metrics.MeanAbsolutePercentageError(),
                                    tf.keras.metrics.MeanAbsoluteError(),
                                    tf.keras.metrics.RootMeanSquaredError()]
                        )
                        model.summary()

                        if(self.callbacks == 1):
                            callbacks = [
                                tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss',
                                    mode='min',
                                    patience=50,
                                    min_delta=0.01, restore_best_weights=True),
                                tf.keras.callbacks.ModelCheckpoint(
                                    filepath=f'{self.path_result_models}{self.model}_{type_dataset}/{code_stock}_{lookback}.h5',
                                    monitor='val_loss',
                                    mode='min',
                                    save_weights_only=False,
                                    save_best_only=True)
                            ]

                        print(f'\n############# Training model {self.model} {type_dataset} {code_stock}...')
                        if(self.is_graph):
                            history = model.fit([x_train, a_train], y_train, epochs=self.epoch,
                                                batch_size=self.batch_size,
                                                validation_data=([x_test, a_test], y_test),
                                                callbacks=(callbacks if self.callbacks == 1 else None))
                        else:
                            history = model.fit(x_train, y_train, epochs=self.epoch,
                                                batch_size=self.batch_size,
                                                validation_data=(x_test, y_test),
                                                callbacks=(callbacks if self.callbacks == 1 else None))

                        print(
                            f'\n############# Prediction & Save model {self.model} {type_dataset} {code_stock}...')
                        
                        if(self.is_graph):
                            y_pred_train = model.predict((x_train, a_train))
                            y_pred_test = model.predict((x_test, a_test))
                        else:
                            y_pred_train = model.predict(x_train)
                            y_pred_test = model.predict(x_test)

                        y_train = pp.back_transform(
                            y_train, self.column_target)
                        y_test = pp.back_transform(
                            y_test, self.column_target)
                        y_pred_train = pp.back_transform(
                            y_pred_train, self.column_target)
                        y_pred_test = pp.back_transform(
                            y_pred_test, self.column_target)

                        score_train = evaluation(
                            y_actual=y_train,
                            y_predict=y_pred_train,
                            model=f'{self.model}_{type_dataset}',
                            filename=f'{code_stock}_{lookback}_train',
                            title_plot=f'Forecasting Stock {code_stock} TRAIN_{lookback}',
                            x_label='Time', y_label='Stock Price',
                            path_datas=self.path_result_datas, 
                            path_evaluations=self.path_result_evaluations, 
                            path_plots=self.path_result_plots)

                        score_test = evaluation(
                            y_actual=y_test,
                            y_predict=y_pred_test,
                            model=f'{self.model}_{type_dataset}',
                            filename=f'{code_stock}_{lookback}_test',
                            title_plot=f'Forecasting Stock {code_stock} TEST_{lookback}',
                            x_label='Time', y_label='Stock Price',
                            path_datas=self.path_result_datas, 
                            path_evaluations=self.path_result_evaluations, 
                            path_plots=self.path_result_plots)

                        print(score_train.measure_performance())
                        print(score_test.measure_performance())

                        del model
                        tf.keras.backend.clear_session()
                        gc.collect()


def main(args):    
    Forecasting(args)
    # pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--code", type=str, default='All',
                        help="All; GOTO; BBCA, BBRI, BMRI; View with command --list_of_code")
    parser.add_argument("--list_of_code", action='store_true',
                        help="Stock Issuer Available")
    parser.add_argument("--type", type=int,
                        default=0, help="List of type: 0: Univariate; 1: Multivariate;")
    parser.add_argument("--lookback", type=str, default='5',
                        help="List of lookback: 5, 10, 20, 50, 100, 200")
    parser.add_argument("--scaler", type=str, default='standard',
                        help="List of scaler: standard; min_max")
    parser.add_argument("--batch_size", type=int,
                        default=128, help="Number of batch size")
    parser.add_argument("--epoch", type=int,
                        default=150, help="Number of epoch")
    parser.add_argument("--callbacks", type=int,
                        default=0, help="Callbacks (Early Stopping & Model Checkpoint) 0: False; 1: True")
    parser.add_argument(
        "--model", type=str, default='TFCNN', help="TFCNN; TFCNNLSTM; TFCNNGRU; \nTFGRU; TFGRUCNN; TFGRULSTM; \nTFLSTM; TFLSTMCNN; TFLSTMGRU; \nTFGCNGRU; TFGCNLSTM")
    args = parser.parse_args()
    main(args)
