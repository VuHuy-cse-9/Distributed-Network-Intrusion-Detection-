from utils import detection_rate, false_alarm_rate, send_model, clone_model_from_local, split_train_test, get_model_dict
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

import pickle

#ARGUMENTS:
parser = argparse.ArgumentParser(description='Distributed Intrusion Detection')
parser.add_argument('--nodeid', default=1.0, type=int, help='Node"s id')
parser.add_argument('--option', default=1, type=int, help='Option for data preprocessing, 0 for category-continous converting, 1 for removing category features.')
args = parser.parse_args()

#Hyper:
N_classifiers = hyper.N_features

#Load data:
X_train, Y_train, labels_idx, labels = get_local_train_dataset(int(args.nodeid), hyper.path_train, hyper.category_features, hyper.skew_features, int(args.option))

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
local_trainer.build() 

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
    
#Send model to global nodes
print(">> Sending model to other nodes ...")

model_dict = get_model_dict(int(args.nodeid), strong_gmms, alphas)

#Send model to other node
send_model(model_dict)
print(">> Model has been sent!")

print(f">> Waiting for receiving other models ...")
local_models, global_alphas = clone_model_from_local(curr_nodeid=nodeid, N_nodes=hyper.N_nodes, N_classifiers=hyper.N_features)
local_models[nodeid] = strong_gmms
global_alphas[nodeid] = alphas

print(f">> Prepare for global dataset ...")
#Select attack sample
# N_global_sample = 100
# X_normal = X_train[Y_train == 1][:N_global_sample]
# X_attack = X_train[Y_train == -1][:N_global_sample]
# N_normal = X_normal.shape[0]
# N_attack = X_attack.shape[0]
# N_global_sample = N_normal + N_attack
# X_global_train = np.concatenate((X_normal, X_attack), axis=0)
# Y_global_label = np.concatenate((np.ones((N_normal, ), dtype=np.int32), 
#                         -1*np.ones((N_attack, ), dtype=np.int32)))

# shuffle_global_index = np.arange(N_global_sample)
# np.random.shuffle(shuffle_global_index)
# X_global_train  = X_global_train[shuffle_global_index]
# Y_global_label = Y_global_label[shuffle_global_index]
X_global_train = X_train
Y_global_label = Y_train

print("=====================GLOBAL DATA SUMMARY=========================")
print(f"Number of normal data in global: {np.sum(Y_global_label == 1)}")
print(f"Number of attack data in global: {np.sum(Y_global_label < 1)}")

#Hyperameter
print(">> Training ....")
global_trainer = PSOSVMTrainer()
global_trainer.build(local_models, global_alphas)
history = global_trainer.fit(X_train, Y_train, X_test, Y_test)

print("Saving model ....")
state = global_trainer.Sg
svc = global_trainer.global_svc
state_dict =  {"state": state.tolist()}
with open(f"checkpoint/global/kdd/state.json", "w") as outfile:
    outfile.write(json.dumps(state_dict))

with open(f"checkpoint/global/kdd/history.json", "w") as outfile:
    outfile.write(json.dumps(history))
    
# save
with open('checkpoint/global/kdd/svm.pkl','wb') as f:
    pickle.dump(svc,f)
    
local_models = global_trainer.local_models
for i in range(len(local_models)):
    #Save model params
    model_dict = get_model_dict(i, local_models[i], global_alphas[i])
    params = json.dumps(model_dict)
    with open(f"checkpoint/global/kdd/model{i}.json", "w") as outfile:
        outfile.write(params)