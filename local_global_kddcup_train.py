from utils import detection_rate, false_alarm_rate, send_model, send_data, clone_model_from_local, split_train_test
import hyper
from visualize import plot_multi_norm_data, plot_global_history
from models.NewOnlineAdaboost import NewOnlineAdaboost
from models.PSOSVM import PSOSVMTrainer
from DataGenerator.DataGenerator import get_data

import numpy as np
import json
from tqdm import tqdm
import argparse
from sklearn.svm import SVC

#ARGUMENTS:
parser = argparse.ArgumentParser(description='Distributed Intrusion Detection')
parser.add_argument('--nodeid', default=1.0, type=int, help='Node"s id')
args = parser.parse_args()

#Hyper:
N_classifiers = hyper.N_features

#Load data:
X_train, Y_train, labels_idx, labels = get_data(hyper.path_train, hyper.category_features)
X_test, Y_test, labels_idx, labels = get_data(hyper.path_test, hyper.category_features)

#Summerize data:
print("=====================LOCAL DATA SUMMARY=========================")
print(f"Train: Number of normal {np.sum(Y_train == 1)}, Number of attack {np.sum(Y_train == -1)}")
print(f"Test: Number of normal {np.sum(Y_test == 1)}, Number of attack {np.sum(Y_test <= -1)}")
    
#Prepare model
print(f">> Prepare model and trainer ....")
local_trainer = NewOnlineAdaboost()
local_trainer.build() 
    
#Visualize:
#print(">> Visualize before training ...")
# gmms = local_trainer.gmms
# means = np.array([gmms[i].means for i in range(N_classifiers)])
# stds = np.array([gmms[i].stds for i in range(N_classifiers)])
# plot_multi_norm_data(X_train, Y_train, means, stds)

print(">> Training local model ...")
local_trainer.fit(X_train, Y_train)
    
print(">> Evaluating local training ...")
dtrs, fars = local_trainer.evaluate(X_test, Y_test) #Detection rate, False alarm rate
print(f"Global result: detection rate {dtrs[0]}, false alarm rate {fars[0]}")
for  index, dtr, far in zip(range(N_classifiers),dtrs, fars):
    print(f"Model feature {index}: detection rate {dtr}, false alarm rate {far}")

#Get trained model
strong_gmms = local_trainer.strong_gmms
alphas = local_trainer.alphas 
    
#Visualize:
#print(">> Visualize after training ...")
# means = np.array([strong_gmms[i].means for i in range(N_classifiers)])
# stds = np.array([strong_gmms[i].stds for i in range(N_classifiers)])
# plot_multi_norm_data(X_train, Y_train, means, stds)
    
#Send model to global nodes
print(">> Sending model to other nodes ...")
nodeid = int(args.nodeid)
model_dict = {}
model_dict["node"] = nodeid
model_dict["alphas"] = alphas.tolist()
for index, gmms in enumerate(strong_gmms):
    model_dict[f"model_{index}"] = gmms.get_parameters()
#print(json.dumps(model_dict))
send_model(model_dict)
print(">> Model has been sent!")

print(f">> Waiting for receiving other models ...")
local_models, global_alphas = clone_model_from_local(curr_nodeid=nodeid, N_nodes=hyper.N_nodes, N_classifiers=hyper.N_features)
local_models[nodeid] = strong_gmms
global_alphas[nodeid] = alphas

print(f">> Prepare for global dataset ...")
#Select attack sample
N_global_sample = 100
X_normal = X_train[Y_train == 1][:N_global_sample]
X_attack = X_train[Y_train == -1][:N_global_sample]
N_normal = X_normal.shape[0]
N_attack = X_attack.shape[0]
N_global_sample = N_normal + N_attack
X_global_train = np.concatenate((X_normal, X_attack), axis=0)
Y_global_label = np.concatenate((np.ones((N_normal, ), dtype=np.int32), 
                        -1*np.ones((N_attack, ), dtype=np.int32)))

shuffle_global_index = np.arange(N_global_sample)
np.random.shuffle(shuffle_global_index)
X_global_train  = X_global_train[shuffle_global_index]
Y_global_label = Y_global_label[shuffle_global_index]

print("=====================GLOBAL DATA SUMMARY=========================")
print(f"Number of normal data in global: {np.sum(Y_global_label == 1)}")
print(f"Number of attack data in global: {np.sum(Y_global_label < 1)}")

#Hyperameter
N_nodes = hyper.N_nodes
Q = hyper.N_states
N_iter = hyper.N_iter
    
print(">> Training ....")
global_trainer = PSOSVMTrainer()
global_trainer.build(local_models, global_alphas)
history = global_trainer.fit(X_train, Y_train, X_test, Y_test)
plot_global_history(history=history, N_iter=N_iter, N_states=Q)