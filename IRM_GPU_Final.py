# -*- coding: utf-8 -*-
#CPU only version

# pytorch mlp for binary classification
import os
import torch.nn as nn
import random
from cupy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.autograd import grad
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.optim import Adam
import time
import torch
import argparse
from sklearn import preprocessing





#values used later in the code, max_epoch is the amount of epochs the program runs for
parser = argparse.ArgumentParser(description='IRM Predictor')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=20)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--erm_weight', type=float, default=1)
parser.add_argument('--irm_weight', type=float, default=1)
parser.add_argument('--hidden_layer', type=bool, default=True)
parser.add_argument('--hidden_dim_size', type=float, default=400)
parser.add_argument('--IRM', type=bool, default=True)
flags = parser.parse_args()


class MLP(nn.Module):
    # define model elements
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

def make_test_environment(path):
    #prepare data as a separate test environment
    
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
    ds = torch.tensor(X).to(device)
    dsanswer = torch.tensor(y).to(device)
    #combine tensors into a list
    ds2 = ds, dsanswer.sum(1 , keepdim = True )
    return ds2

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
#    print("Val Length:", val_len, "traing length:", train_len)
    
    combined_data_train = (combined_data[0][val_len:train_len], combined_data[1][val_len:train_len])
    combined_data_val = (combined_data[0][:val_len], combined_data[1][:val_len])
#    print("ds2_train:", len(ds2_train[0]),  "ds2_val:", len(ds2_val[0]))
    return combined_data_train, combined_data_val

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
    optimizer = Adam(model.parameters(), lr=flags.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #training loop
    for epoch in range(flags.max_epoch):
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
    return ((preds - y) < 1e-2).mean() 

# evaluate the model accuracy
def evaluate_model(test_env, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_env):
 
        yhat = model(inputs)
        # retrieve numpy array
#        yhat = yhat.to("cpu")
        yhat = yhat.detach()
#        targets = targets
        actual = targets
        actual = actual.reshape((len(actual), 1))
#        print("actual:", actual)
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)

#        print("pred:", predictions)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    acc = mean_accuracy(predictions, actuals)
    acc = acc.mean()

    return acc

#create a table to output the results nicely
def complete_table(avg_train_acc, avg_test_acc, avg_in_distribution_test_acc):
    rows = [['learning_rate', 'IRM', 'Hidden_Layer', 'Hidden_layer_Size', 'max_epochs', 'erm_weight', 'irm_weight', 'training_acc', 'I_D_T_A', 'test_acc']]
    for n in range (0, 1):
        rows.append((str(flags.lr), str(flags.IRM), str(flags.hidden_layer), str(flags.hidden_dim_size), \
     str(flags.max_epoch), str(flags.erm_weight), str(flags.irm_weight), str(avg_train_acc), str(avg_in_distribution_test_acc), str(avg_test_acc)))
            
    def pretty_table(rows, column_count, column_spacing=4):
        aligned_columns = []
        for column in range(column_count):
            column_data = list(map(lambda row: row[column], rows))
            aligned_columns.append((max(map(len, column_data)) + column_spacing, column_data))
                    
        for row in range(len(rows)):
            aligned_row = map(lambda x: (x[0], x[1][row]), aligned_columns)
            yield ''.join(map(lambda x: x[1] + ' ' * (x[0] - len(x[1])), aligned_row))
                        
    for line in pretty_table(rows, 10):
        print (line)

#sets the device to gpu if available
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
print("device:", device)
#initializing accuracy variables
total_in_distribution_test_acc = 0
avg_in_distribution_test_acc = 0
total_test_acc = 0
total_train_acc = 0
#start time and file name
start_time = time.time()
file_name = (os.path.basename(__file__))
#hyperparameter printing and file name printing
#print("File name:", file_name, "Restarts:", flags.n_restarts,\
#"Learning rate:", flags.lr, "max_epochs:", flags.max_epoch, \
#"ERM weight:", flags.erm_weight, "IRM weight:", flags.irm_weight)
#change file directory to data file
#print(os.path.dirname(os.path.realpath(__file__)))
os.chdir('Wirbel_Zeller_2019_Fecal_Data')

if not flags.hidden_layer:
    flags.hidden_dim_size = 'N/A'

# number of restarts
for b in range(0, flags.n_restarts):
#    print("Run:", b + 1)

    # define the network
    model = MLP(849).to(device)
        
    # data file paths
    test_files = ["France.csv"]
    training_files = ["Germany.csv", "USA.csv", "Austria.csv", "China.csv"]
        
    #create environments
    training_environments, in_distribution_test_set, test_environments = load_data(training_files, test_files)
        
    #training the model
    train_model(training_environments, model)
        
    #storing and averaging accuracy
    in_distribution_test_acc = evaluate_model(in_distribution_test_set, model)
    total_in_distribution_test_acc += in_distribution_test_acc
    train_acc = evaluate_model(training_environments, model)
    total_train_acc += train_acc
    test_acc = evaluate_model(test_environments, model)
    total_test_acc += test_acc
#    complete_table(train_acc, test_acc, in_distribution_test_acc)
    
#calculating average accuracy
avg_train_acc = total_train_acc/flags.n_restarts
avg_test_acc = total_test_acc/flags.n_restarts
avg_in_distribution_test_acc = total_in_distribution_test_acc/flags.n_restarts

#calculating run time
end_time = time.time()
run_time = end_time-start_time


#printing results
print("Run time:", run_time)
complete_table(avg_train_acc, avg_test_acc, avg_in_distribution_test_acc)