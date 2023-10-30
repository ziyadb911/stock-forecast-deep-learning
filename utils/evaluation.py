import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report


class evaluation:
    def __init__(self, y_actual, y_predict, model, filename, show_plot=False, 
                 title_plot='', x_label='', y_label='', path_datas='', path_evaluations='', path_plots=''):
        self.y_actual = y_actual
        self.y_predict = y_predict
        self.show_plot = show_plot
        self.title_plot = title_plot
        self.x_label = x_label
        self.y_label = y_label
        self.filename = filename

        self.path_y_data = f'{path_datas}/{model}//'
        self.evaluation = f'{path_evaluations}/{model}//'
        self.plot = f'{path_plots}/{model}//'

        self.create_folder(f'{self.path_y_data}')
        self.create_folder(f'{self.evaluation}')
        self.create_folder(f'{self.plot}')
        
    def create_folder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
    
    def check_files(self, pathFile):
        if os.path.isfile(pathFile):
            os.remove(pathFile)

    # differencing time series
    def difference(self, before, after, interval=1):
        diff = list()
        level = list()
        for i in range(interval, len(before)):
            value = after[i] - before[i - interval]
            diff.append(value)
            level.append(0 if value <= 0 else 1)
        return diff, level

    def accuracy_mean_absolute_percentage_error(self, y_true, y_pred):
        y_true_diff, y_true_class = self.difference(
            y_true, y_true)
        y_pred_diff, y_pred_class = self.difference(
            y_true, y_pred)

        if len(set(y_true_class)) == 2:
            class_report = classification_report(
                y_true_class, y_pred_class, target_names=['DOWN', 'UP'], output_dict=True)
        elif set(y_true_class) == set(y_pred_class):
            class_report = {'accuracy':1}
        elif set(y_true_class) != set(y_pred_class):
            class_report = {'accuracy':0}
            
        # save classification report
        if(len(self.filename) > 0):
            pathFile = f'{self.evaluation}{self.filename}_classification.json'
            self.check_files(pathFile)
            with open(pathFile, 'w') as fp:
                json.dump(class_report, fp,  indent=4)

        if len(set(y_true_class)) == 2:
            print(classification_report(y_true_class, y_pred_class, target_names=['DOWN', 'UP']))

        mape = mean_absolute_percentage_error(y_true, y_pred)
        acmape = ((1-class_report['accuracy'])*2) + mape
        return acmape

    def measure_performance(self):
        y_data = np.transpose(
            np.array([self.y_actual.flatten(), self.y_predict.flatten()]))
        df_y = pd.DataFrame(y_data, columns=["y_actual", "y_predict"])

        # save data actual and predict
        if(len(self.filename) > 0):
            pathFile = f'{self.path_y_data}{self.filename}_dfy.csv'
            self.check_files(pathFile)
            df_y.to_csv(pathFile)

        # save plot actual and predict
        plt.plot(self.y_actual, color='blue', alpha=0.7, linewidth=0.7)
        plt.plot(self.y_predict, color='orange', alpha=0.7, linewidth=0.7)
        plt.title(self.title_plot)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend(['Actual', 'Forecast'])

        if(len(self.filename) > 0):
            pathFile = f'{self.plot}{self.filename}_predict'
            self.check_files(pathFile)
            plt.savefig(pathFile)
        if(self.show_plot):
            plt.show()
        plt.close()

        # calculate errors
        r2 = r2_score(self.y_actual, self.y_predict)
        rmse = mean_squared_error(self.y_actual, self.y_predict, squared=False)
        mse = mean_squared_error(self.y_actual, self.y_predict, squared=True)
        mape = mean_absolute_percentage_error(self.y_actual, self.y_predict)
        acmape = self.accuracy_mean_absolute_percentage_error(
            self.y_actual, self.y_predict)

        score = {
            "R2": r2,
            "RMSE": rmse,
            "MSE": mse,
            "MAPE": mape,
            "ACMAPE": acmape
        }
        
        # save erros
        if(len(self.filename) > 0):
            pathFile = f'{self.evaluation}{self.filename}_score.json'
            self.check_files(pathFile)
            with open(pathFile, 'w') as fp:
                json.dump(score, fp,  indent=4)

        return score
