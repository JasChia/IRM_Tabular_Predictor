# -*- coding: utf-8 -*-
#CPU only version

# pytorch mlp for binary classification
import os
import torch.nn as nn
import random
from cupy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from torch.autograd import grad
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.optim import Adam
import time
import torch
import argparse
from sklearn import preprocessing

#values used later in the code
parser = argparse.ArgumentParser(description='IRM Predictor')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--erm_weight', type=float, default=1)
parser.add_argument('--irm_weight', type=float, default=1)
parser.add_argument('--hidden_layer', type=bool, default=True)
parser.add_argument('--hidden_dim_size', type=int, default=400)
parser.add_argument('--IRM', type=bool, default=True)
parser.add_argument('--test_files', type=str, default="WirbelZeller_2019_CRC_France.csv")
parser.add_argument('--training_files', type=str, default="WirbelZeller_2019_CRC_Germany.csv,WirbelZeller_2019_CRC_USA.csv,WirbelZeller_2019_CRC_Austria.csv,WirbelZeller_2019_CRC_China.csv")
flags = parser.parse_args()

#defining the model
class MLP(nn.Module):
    # define model layers
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        if flags.hidden_layer:
            layer1 = nn.Linear(n_inputs, flags.hidden_dim_size)
            layer2 = nn.Linear(flags.hidden_dim_size, 1)        
            kaiming_uniform_(layer1.weight, nonlinearity='relu')
            xavier_uniform_(layer2.weight)
            self.main_ = nn.Sequential(layer1, nn.ReLU(), layer2, nn.Sigmoid())
        else:
            layer1 = nn.Linear(n_inputs, 1)
            xavier_uniform_(layer1.weight)
            self.main_ = nn.Sequential(layer1, nn.Sigmoid())

    def forward(self, X):
        X = self.main_(X)
        return X

def model_size(path):
    data = read_csv(path, header=None)
    return(len(data.columns) - 1)

#prepare data as a separate test environment
def make_test_environment(path):
    data = read_csv(path, header=None)
    X = data.values[:, :-1]
    #normalize values
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    #prepare targets
    y = data.values[:, -1]
    y = LabelEncoder().fit_transform(y)
    y = y.astype('float32')
    y = y.reshape((len(y), 1))
    
    #ensure proper data type
    X = X.astype('float32')
    y = y.astype('float32')
    #convert and return as tensors
    input_data = torch.tensor(X).to(device)
    target_data = torch.tensor(y).to(device)
    #combine tensors into a list
    combined_data = input_data, target_data.sum(1 , keepdim = True )
    #Shuffle data
    rng_state = random.getstate()
    random.shuffle(combined_data[0])
    random.setstate(rng_state)
    random.shuffle(combined_data[1])
    return combined_data

#Prepare data as a training environment
def make_training_environment(path):
    #grab data
    data = read_csv(path, header=None)
    X = data.values[:, :-1]
    #print(X)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    y = data.values[:, -1]
    y = LabelEncoder().fit_transform(y)
    y = y.astype('float32')
    y = y.reshape((len(y), 1))
    
    #ensure proper data type
    X = X.astype('float32')
    y = y.astype('float32')
    #convert data into tensors
    input_data = torch.tensor(X).to(device)
    target_data = torch.tensor(y).to(device)
    #combine tensors
    combined_data = input_data, target_data.sum (1 , keepdim = True )
    #shuffle data
    rng_state = random.getstate()
    random.shuffle(combined_data[0])
    random.setstate(rng_state)
    random.shuffle(combined_data[1])
    #separate 10% of data for in distribution test data
    val_len = len(combined_data[0]) * 0.1
    val_len = int(val_len)
    train_len = len(combined_data[0]) - val_len
    train_len = int(train_len)
    
    combined_data_train = (combined_data[0][val_len:train_len], combined_data[1][val_len:train_len])
    combined_data_val = (combined_data[0][:val_len], combined_data[1][:val_len])
    return combined_data_train, combined_data_val

#Loading data into the proper formats
def load_data(training_files, test_files):
    training_environments = []
    test_environments = []
    in_distribution_test_set = []
        
    #creating test and training environments
    for file in test_files:
        test_environments.append(make_test_environment(file))

    for file in training_files:
        temp_train, temp_test = make_training_environment(file)
        training_environments.append(temp_train)
        in_distribution_test_set.append(temp_test)
    return training_environments, in_distribution_test_set, test_environments

#Invariant risk minimization penalty
def irm_penalty(logits, y):

    scale = torch.nn.Parameter (torch.Tensor ([1.0])).to(device)
    _BCELoss = nn.BCELoss(reduction ="none")
    loss_1 = _BCELoss(logits * scale, y)
    grad_1 = grad(loss_1.mean(), scale, create_graph=True)[0]
    result = torch.sum(grad_1 ** 2)
    return result

# train the model
def train_model(training_env, model):
    #set up the optimizer and loss function
    criterion = nn.BCELoss(reduction ="none")
    optimizer = Adam(model.parameters(), lr=flags.learning_rate)
    #training loop
    for epoch in range(flags.epoch):
        error = 0
        penalty = 0
        for inputs, targets in training_env:
            #model prediction    
            yhat = model(inputs)
            #ERM loss function
            error_e = criterion(yhat, targets)
            #IRM loss function
            if flags.IRM:
                penalty += irm_penalty(yhat, targets)

            error += error_e.mean()
                
        loss = flags.erm_weight * error + penalty * flags.irm_weight
        # update model weights
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()

def mean_accuracy(logits, y):
    preds = (logits > 0.)
    return (abs(preds - y) < 1e-2).mean() 

# evaluate the model accuracy
def evaluate_model(test_env, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_env):
 
        yhat = model(inputs)

        yhat = yhat.detach()
        actual = targets
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)

    predictions, actuals = vstack(predictions), vstack(actuals)
    acc = mean_accuracy(predictions, actuals)
    return acc

#create a table to output the results nicely
def complete_table(avg_train_acc, avg_test_acc, avg_in_distribution_test_acc):
    rows = [['learning Rate', 'IRM', 'Hidden Layer', 'Hidden layer Size', 'Epochs', 'ERM weight', 'IRM weight', 'Training acc', 'IDTA', 'OOD test acc',\
             'Test Files', "Training Files"]]
    for n in range (0, 1):
        rows.append((str(flags.learning_rate), str(flags.IRM), str(flags.hidden_layer), str(flags.hidden_dim_size), \
     str(flags.epoch), str(flags.erm_weight), str(flags.irm_weight), str(avg_train_acc), str(avg_in_distribution_test_acc), str(avg_test_acc),\
         str(flags.test_files), str(flags.training_files)))
            
    def pretty_table(rows, column_count, column_spacing=4):
        aligned_columns = []
        for column in range(column_count):
            column_data = list(map(lambda row: row[column], rows))
            aligned_columns.append((max(map(len, column_data)) + column_spacing, column_data))
                    
        for row in range(len(rows)):
            aligned_row = map(lambda x: (x[0], x[1][row]), aligned_columns)
            yield ''.join(map(lambda x: x[1] + ' ' * (x[0] - len(x[1])), aligned_row))
                        
    for line in pretty_table(rows, 12):
        print (line)

#sets the device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("file name:", (os.path.basename(__file__)), "device:", device)
#initializing accuracy variables
total_in_distribution_test_acc = 0
avg_in_distribution_test_acc = 0
total_test_acc = 0
total_train_acc = 0
#start time and data file directory
start_time = time.time()
os.chdir('Data')

if not flags.hidden_layer:
    flags.hidden_dim_size = 'N/A'

# number of restarts
for b in range(0, flags.n_restarts):

    temp_time = time.time()
    
    # data file paths
    test_files = flags.test_files.split(",")
    training_files = flags.training_files.split(",")
    #create environments
    training_environments, in_distribution_test_set, test_environments = load_data(training_files, test_files)
    
    # define the network
    model = MLP(model_size(training_files[0])).to(device)    
    
    #training the model
    train_model(training_environments, model)
        
    #storing and averaging accuracy
    in_distribution_test_acc = evaluate_model(in_distribution_test_set, model)
    total_in_distribution_test_acc += in_distribution_test_acc
    train_acc = evaluate_model(training_environments, model)
    total_train_acc += train_acc
    test_acc = evaluate_model(test_environments, model)
    total_test_acc += test_acc
    temp_end_time = time.time()
    temp_run_time = temp_end_time-temp_time

#calculating average accuracy
avg_train_acc = total_train_acc/flags.n_restarts
avg_test_acc = total_test_acc/flags.n_restarts
avg_in_distribution_test_acc = total_in_distribution_test_acc/flags.n_restarts

#calculating run time
end_time = time.time()
run_time = end_time-start_time


#printing results
complete_table(avg_train_acc, avg_test_acc, avg_in_distribution_test_acc)
