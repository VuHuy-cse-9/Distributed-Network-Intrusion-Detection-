from utils import detection_rate, false_alarm_rate, send_model, clone_model_from_local, get_model_dict
import hyper
from visualize import plot_multi_norm_data, plot_global_history
from models.NewOnlineAdaboost import NewOnlineAdaboost
from models.PSOSVM import PSOSVMTrainer
from DataGenerator.DataGenerator import get_local_train_dataset

import numpy as np
import json
from tqdm import tqdm
import argparse
from sklearn.svm import SVC

import os.path

import pickle

#ARGUMENTS:
parser = argparse.ArgumentParser(description='Distributed Intrusion Detection')
parser.add_argument('--nodeid', default=1.0, type=int, help='Node"s id')
parser.add_argument('--option', default=1, type=int, help='Option for data preprocessing, 0 for category-continous converting, 1 for removing category features.')
args = parser.parse_args()

#Hyper:
N_classifiers = hyper.n_features
nodeid = int(args.nodeid)

#Load data:
X_train, Y_train, labels_idx, labels, _ = get_local_train_dataset(nodeid, hyper.path_train, hyper.category_features, hyper.skew_features, int(args.option))

#Summerize data:
print("=====================LOCAL DATA SUMMARY=========================")
print(f"Train: Number of normal {np.sum(Y_train == 1)}, Number of attack {np.sum(Y_train <= -1)}")
print(f"normal: {Y_train[Y_train == 1].shape[0]}")
print(f"neptune: {Y_train[Y_train == hyper.attack_global_train['neptune.']].shape[0]}")
print(f"snmpgetattack: {Y_train[Y_train == hyper.attack_global_train['snmpgetattack.']].shape[0]}")
print(f"mailbomb: {Y_train[Y_train == hyper.attack_global_train['mailbomb.']].shape[0]}")
print(f"smurf: {Y_train[Y_train == hyper.attack_global_train['smurf.']].shape[0]}")
    
#Prepare model
print(f">> Prepare model and trainer ....")
local_trainer = NewOnlineAdaboost()
local_trainer.build(hyper.n_labels, hyper.n_features) 

print(">> Training local model ...")
local_trainer.fit(X_train, Y_train)

#Get trained model
strong_gmms = local_trainer.strong_gmms
alphas = local_trainer.alphas 
    
#Send model to global nodes
print(">> Sending model to other nodes ...")

model_dict = get_model_dict(nodeid, strong_gmms, alphas)

#Send model to other node
send_model(model_dict)
print(">> Model has been sent!")

print(f">> Waiting for receiving other models ...")
print(f">> Waiting for receiving other models ...")
strong_gmms_list, global_alphas = clone_model_from_local(curr_nodeid=nodeid, N_nodes=hyper.N_nodes, N_classifiers=hyper.n_features)
strong_gmms_list[nodeid] = strong_gmms
global_alphas[nodeid] = alphas
local_models = []
for i in range(hyper.N_nodes):
    local_model = NewOnlineAdaboost()
    local_model.set_params(strong_gmms_list[i], global_alphas[i], hyper.n_features)
    local_models.append(local_model)
local_models = np.array(local_models)

print(f">> Prepare for global dataset ...")
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
print(f"Number of train samples {X_global_train.shape[0]}, Number of features: {X_global_train.shape[1]}")
print(f"Number of test samples {X_global_test.shape[0]}, Number of features: {X_global_test.shape[1]}")
print(f"TRAIN: Number of normal {np.sum(Y_global_train == 1)}, Number of attack {np.sum(Y_global_train <= -1)}")
print(f"Number of labels: {np.unique(Y_global_train).shape[0]}")
print(f"Number of : {len(Y_global_train[Y_global_train == -1])}")
print(f"Number of : {len(Y_global_train[Y_global_train == -2])}")
print(f"Number of : {len(Y_global_train[Y_global_train == -3])}")
print(f"Number of : {len(Y_global_train[Y_global_train == -4])}")
print(f"Number of : {len(Y_global_train[Y_global_train == -5])}")
print("-------------------------------------------------------------")
print(f"TEST: Number of normal {np.sum(Y_global_test == 1)}, Number of attack {np.sum(Y_global_test <= -1)}")
print(f"Number of labels: {np.unique(Y_global_test).shape[0]}")
print(f"Number of : {len(Y_global_test[Y_global_test == -1])}")
print(f"Number of : {len(Y_global_test[Y_global_test == -2])}")
print(f"Number of : {len(Y_global_test[Y_global_test == -3])}")
print(f"Number of : {len(Y_global_test[Y_global_test == -4])}")
print(f"Number of : {len(Y_global_test[Y_global_test == -5])}")
print("=================================================================")

#Hyperameter
print(">> Training ....")
global_trainer = PSOSVMTrainer()
global_trainer.build(local_models)
history = global_trainer.fit(X_global_train, Y_global_train, X_global_test, Y_global_test)

print("Saving model ....")
state = global_trainer.Sg
svc = global_trainer.global_svc
state_dict =  {"state": state.tolist()}

if os.path.exists(f"checkpoint/global/kdd/state.json"):
    with open(f"checkpoint/global/kdd/state.json", "w") as outfile:
        outfile.write(json.dumps(state_dict))
else:
    with open(f"checkpoint/global/kdd/state.json", "x") as outfile:
        outfile.write(json.dumps(state_dict))
        
if os.path.exists(f"checkpoint/global/kdd/history.json"):
    with open(f"checkpoint/global/kdd/history.json", "w") as outfile:
        outfile.write(json.dumps(history))
else:
    with open(f"checkpoint/global/kdd/history.json", "x") as outfile:
        outfile.write(json.dumps(history))
    
# save

if os.path.exists(f"checkpoint/global/kdd/svm.pkl"):
    with open(f"checkpoint/global/kdd/svm.pkl", "wb") as outfile:
        pickle.dump(svc,outfile)
else:
    with open(f"checkpoint/global/kdd/svm.pkl", "xb") as outfile:
        pickle.dump(svc,outfile)

    
local_models = global_trainer.local_models
for i in range(len(local_models)):
    #Save model params
    model_dict = get_model_dict(i, local_models[i].strong_gmms, local_models[i].alphas)
    params = json.dumps(model_dict)
    if os.path.exists(f"checkpoint/global/kdd/model{i}.json"):
        with open(f"checkpoint/global/kdd/model{i}.json", "w") as outfile:
            outfile.write(params)
    else:
        with open(f"checkpoint/global/kdd/model{i}.json", "x") as outfile:
            outfile.write(params)