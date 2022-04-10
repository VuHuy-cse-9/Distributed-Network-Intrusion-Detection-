from DataGenerator.DataGenerator import get_data
import hyper
from utils import detection_rate, false_alarm_rate, send_model, clone_model_from_local, get_model_dict
from DataGenerator.DataGenerator import load_fake_data
from visualize import plot_multi_norm_data, plot_global_history, roc_curve_plot
from models.NewOnlineAdaboost import NewOnlineAdaboost
from models.PSOSVM import PSOSVMTrainer

import numpy as np
import json
from tqdm import tqdm
import argparse
from sklearn.svm import SVC

import pickle

#ARGUMENTS:
parser = argparse.ArgumentParser(description='Distributed Intrusion Detection')
parser.add_argument('--nodeid', default=1.0, type=int, help='Node"s id')
args = parser.parse_args()

#Load fake data:
N_labels, N_train, N_test = 2, 1000, 500
print(">> Loading dataset ...")
X_train, X_test, Y_train, Y_test = load_fake_data(N_train, N_test, hyper.n_features)
N_train = N_train * 2 #JUST FOR TEMPORARY
N_test = N_test * 2


#Summerize data:
print("=====================LOCAL DATA SUMMARY=========================")
print(f"Number of samples: {X_train.shape[0]}")
print(f"Train: Number of normal {np.sum(Y_train == 1)}, Number of attack {np.sum(Y_train <= -1)}")
print(f"Test: Number of normal {np.sum(Y_test == 1)}, Number of attack {np.sum(Y_test == -1)}")
    
#Prepare model
print(f">> Prepare model and trainer ....")
local_trainer = NewOnlineAdaboost()
local_trainer.build(n_labels=hyper.n_labels, n_features=hyper.n_features) 

print(">> Training local model ...")
local_trainer.fit(X_train, Y_train)
    
print(">> Evaluating local training ...")
y_scores = local_trainer.predict_score(X_test)
y_preds = np.sign(y_scores)

dtr = detection_rate(Y_test, y_preds)
far = false_alarm_rate(Y_test, y_preds)

print("===========LOCAL RESULT=============")
print(f">> dtr: {dtr * 100}%, far: {far * 100}%")
roc_curve_plot(Y_test, y_preds)

#Get trained model
strong_gmms = local_trainer.strong_gmms
alphas = local_trainer.alphas 
    
#Visualize:
# print(">> Visualize after training ...")
# means = np.array([strong_gmms[i].means for i in range(N_classifiers)])
# stds = np.array([strong_gmms[i].stds for i in range(N_classifiers)])
# plot_multi_norm_data(X_train, Y_train, means, stds)
    
#Send model to global nodes
print(">> Sending model to other nodes ...")
nodeid = int(args.nodeid)
model_dict = get_model_dict(nodeid, strong_gmms, alphas)
send_model(model_dict)
print(">> Model has been sent!")

print(f">> Waiting for receiving other models ...")
strong_gmms_list, global_alphas = clone_model_from_local(curr_nodeid=nodeid, N_nodes=hyper.N_nodes, N_classifiers=hyper.n_features)
strong_gmms_list[nodeid] = strong_gmms
global_alphas[nodeid] = alphas
local_models = []
for i in range(hyper.N_nodes):
    local_models = NewOnlineAdaboost()
    local_models.set_params(strong_gmms_list[i], global_alphas[i], hyper.n_features)
local_models = np.array(local_models)

def get_random_index_array(N):
    array_index = np.arange(N)
    np.random.shuffle(array_index)
    return array_index

def select_global_data(X, Y, n_attack, n_samples):
    shuffle_index = get_random_index_array(X[Y == 1].shape[0])
    X_select = X[Y == 1][shuffle_index][:n_samples]
    Y_select = Y[Y == 1][shuffle_index][:n_samples]
    for i in range(n_attack):
        attack_index = -1*i - 1
        shuffle_index = get_random_index_array(X[Y == attack_index].shape[0])
        X_attack = X[Y == attack_index][shuffle_index][:n_samples]
        Y_attack = Y[Y == attack_index][shuffle_index][:n_samples]
        X_select = np.concatenate((X_select, X_attack), axis=0)
        Y_select = np.concatenate((Y_select, Y_attack), axis=0)
    shuffle_index = get_random_index_array(X_select.shape[0])
    X_select = X_select[shuffle_index]
    Y_select = Y_select[shuffle_index]
    return X_select, Y_select
    

N_train_global_sample, N_test_global_sample = 200, 100
X_global_train, Y_global_train = select_global_data(X_train, Y_train, hyper.n_labels - 1, N_train_global_sample)
X_global_test, Y_global_test = select_global_data(X_train, Y_train, hyper.n_labels - 1, N_test_global_sample)

print("=====================GLOBAL DATA SUMMARY=========================")
print(f"Number of samples: {X_global_train.shape[0]}")
print(f"Number of features: {X_global_train.shape[1]}")
print(f"Labels: {np.unique(Y_global_label)}")
print(f"Number of normal data in global: {np.sum(Y_global_label == 1)}")
print(f"Number of attack data in global: {np.sum(Y_global_label <= -1)}")

#Hyperameter
N_nodes = hyper.N_nodes
Q = hyper.N_states
N_iter = hyper.N_iter
    
print(">> Training ....")
global_trainer = PSOSVMTrainer()
global_trainer.build(local_models)
history = global_trainer.fit(X_global_train, Y_global_train, X_global_test, Y_global_test)

local_models = global_trainer.local_models

print("Saving model ....")
state = global_trainer.Sg
svc = global_trainer.global_svc
state_dict =  {"state": state.tolist()}
with open(f"checkpoint/global/rand/state.json", "w") as outfile:
    outfile.write(json.dumps(state_dict))

with open(f"checkpoint/global/rand/history.json", "w") as outfile:
    outfile.write(json.dumps(history))
    
# save
with open('checkpoint/global/rand/svm.pkl','wb') as f:
    pickle.dump(svc,f)
    
local_models = global_trainer.local_models
for i in range(len(local_models)):
    #Save model params
    model_dict = get_model_dict(i, local_models[i].strong_gmms, local_models[i].alphas)
    params = json.dumps(model_dict)
    with open(f"checkpoint/global/rand/model{i}.json", "w") as outfile:
        outfile.write(params)
        
y_preds = global_trainer.predict(X_test)
dtr = detection_rate(Y_test, y_preds)
far = false_alarm_rate(Y_test, y_preds)

print("============RESULT===========")
print(f">> dtr: {dtr * 100}%, far: {far * 100}%")