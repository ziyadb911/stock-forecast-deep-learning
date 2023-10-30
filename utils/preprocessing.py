import os
import pandas as pd
import numpy as np
import tensorflow as tf
import networkx as nx

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class preprocessing:
    def __init__(self,
                 scaler='standard',
                 model='',
                 filename='',
                 path_datas=''):
        self.scaler_type = scaler
        self.model_name = model
        self.filename = filename
        self.column_y = ''
        if scaler == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.path_y_data = f'{path_datas}/{model}//'
        self.create_folder(f'{self.path_y_data}')

    def create_folder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def check_files(self, pathFile):
        if os.path.isfile(pathFile):
            os.remove(pathFile)

    def restore_dataset(self, dataset):
        inputs = []
        targets = []
        for batch in tqdm(dataset, desc='Restore dataset sequence'):
            inputs_a, targets_a = batch
            inputs_a = inputs_a.numpy()
            targets_a = targets_a.numpy()
            for i in range(0, len(inputs_a)):
                inputs.append(inputs_a[i])
                targets.append(targets_a[i][0])

        inputs = np.array(inputs)
        targets = np.array(targets)

        return inputs, targets

    def dataset_sequance(self, feature, target):
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            feature,
            target,
            sequence_length=self.lookback,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=self.batch_size,
        )
        return self.restore_dataset(dataset)    

    def dataset_cleaning(self, dataset):
        # cleaning column, remove column if NA > 5%
        column_remove = []
        for (columnName, columnData) in tqdm(dataset.items(), desc="Dataset cleaning"):
            percentage_na = dataset[columnName].isna(
            ).sum()/len(dataset)
            percentage_zero = (
                dataset[columnName] == 0).sum()/len(dataset)
            if(percentage_na > 0.05 or percentage_zero > 0.5):
                column_remove.append(columnName)
        dataset = dataset.drop(columns=column_remove)

        # replace inf value with nan
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset = dataset.fillna(0)  # replace nan value with 0
        return dataset
    
    def dataset_univariate(self,
                           dataset,
                           lookback=10,
                           prediction_days=30,
                           batch_size=128):
        self.dataset = dataset
        self.lookback = lookback
        self.prediction_days = prediction_days
        self.batch_size = batch_size

        self.dataset = self.dataset[self.dataset.notna()]

        # save dataset
        if(len(self.filename) > 0):
            pathFile = f'{self.path_y_data}{self.filename[:4]}_dfx.csv'
            self.check_files(pathFile)
            self.dataset.to_csv(pathFile)

        train_data = self.dataset[0: (
            len(self.dataset) - self.prediction_days + 1)]
        test_data = self.dataset[len(train_data) - self.lookback - 1:]

        train_data = self.scaler.fit_transform(
            train_data.values.reshape(-1, 1))
        test_data = self.scaler.transform(
            test_data.values.reshape(-1, 1))

        x_train, y_train = self.dataset_sequance(
            train_data, train_data[self.lookback:])
        x_test, y_test = self.dataset_sequance(
            test_data, test_data[self.lookback:])

        return x_train, x_test, y_train, y_test
    
    def dataset_multivariate(self,
                             dataset,
                             column_y,
                             lookback=10,
                             prediction_days=30,
                             batch_size=128):
        self.dataset = self.dataset_cleaning(dataset)
        self.lookback = lookback
        self.prediction_days = prediction_days
        self.batch_size = batch_size
        self.column_y = column_y
        columns = self.dataset.columns

        self.dataset = self.dataset[self.dataset.notna()]

        # save dataset
        if(len(self.filename) > 0):
            pathFile = f'{self.path_y_data}{self.filename[:4]}_dfx.csv'
            self.check_files(pathFile)
            self.dataset.to_csv(pathFile)

        train_data = self.dataset[0: (
            len(self.dataset) - self.prediction_days + 1)]
        test_data = self.dataset[len(train_data) - self.lookback - 1:]

        train_data = self.scaler.fit_transform(train_data)
        test_data = self.scaler.transform(test_data)

        train_data = pd.DataFrame(train_data, columns=columns)
        test_data = pd.DataFrame(test_data, columns=columns)

        y_train = train_data[self.column_y][self.lookback:].reset_index(
            drop=True)
        y_test = test_data[self.column_y][self.lookback:].reset_index(
            drop=True)

        x_train, y_train = self.dataset_sequance(
            train_data,
            y_train.values.reshape(-1, 1)
        )
        x_test, y_test = self.dataset_sequance(
            test_data,
            y_test.values.reshape(-1, 1)
        )

        return x_train, x_test, y_train, y_test
    
    def dataset_graph(self,
                      dataset,
                      column_y,
                      lookback=10,
                      prediction_days=30,
                      batch_size=128):
        
        x_train, x_test, y_train, y_test = self.dataset_multivariate(dataset, column_y, lookback, prediction_days, batch_size)
        a_train, g_train = self.build_graph(x_train)
        a_test, g_test = self.build_graph(x_test)
        
        return x_train, x_test, y_train, y_test, a_train, a_test

    def build_graph(self, dataset):
        # Compute pairwise correlations between feature
        dataset = dataset.transpose(0, 2, 1)
        adjacency_matrix = []
        graph_adjacency_matrix = []
        for i in range(dataset.shape[0]):
            corr_matrix = np.corrcoef(dataset[i].T)
            np.fill_diagonal(corr_matrix, 0)  # set diagonal to 0
            
            G = nx.Graph()
            for i in range(corr_matrix.shape[0]):
                for j in range(i+1, corr_matrix.shape[1]):
                    # weight = 0 if corr_matrix[i,j] < 0.9 else corr_matrix[i,j]
                    weight = corr_matrix[i,j]                
                    G.add_edge(i, j, weight=weight)
                    
            # Convert graph to adjacency matrix
            A = nx.adjacency_matrix(G)
            
            adjacency_matrix.append(A.todense())
            graph_adjacency_matrix.append(A)
        
        adjacency_matrix = np.array(adjacency_matrix)        
        
        return adjacency_matrix, graph_adjacency_matrix


    def back_transform(self, data, column_target=''):
        if(len(self.scaler.get_feature_names_out()) == 1):
            # univariate
            return self.scaler.inverse_transform(data.reshape(-1, 1))
        else:
            # multivariate
            column_names = self.scaler.get_feature_names_out()
            dummy = pd.DataFrame(
                np.zeros((len(data), len(column_names))), columns=column_names)
            dummy[column_target] = data
            dummy = pd.DataFrame(
                self.scaler.inverse_transform(dummy), columns=column_names)
            return dummy[column_target].values
