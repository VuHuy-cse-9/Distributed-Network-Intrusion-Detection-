from models.NewOnlineAdaboost import NewOnlineAdaboost
from models.PSOSVM import PSOSVMTrainer
from DataGenerator.DataGenerator import get_data
import hyper
from utils import detection_rate, false_alarm_rate, send_model, clone_model_from_local, get_model_dict
from visualize import plot_multi_norm_data, plot_global_history

import json
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.svm import SVC
import os.path


#ARGUMENTS:
parser = argparse.ArgumentParser(description='Distributed Intrusion Detection')
parser.add_argument('--nodeid', default=1, type=int, help='Node"s id')
parser.add_argument('--option', default=1, type=int, help='Option for data preprocessing, 0 for category-continous converting, 1 for removing category features.')
args = parser.parse_args()

#Load data:
X_train, Y_train, labels_idx, labels, _ = get_data(hyper.path_train, hyper.category_features, None, int(args.option))

#For testing only
# X_train_normal = X_train[Y_train == 1][:8000, :]
# X_train_1 = X_train[Y_train == -1][:15000, :]
# X_train_2 = X_train[Y_train == -2][:8000, :]
# X_train_3 = X_train[Y_train == -3][:8000, :]
# X_train_4 = X_train[Y_train == -4][:8000, :]
# Y_train_normal = Y_train[Y_train == 1][:8000]
# Y_train_1 = Y_train[Y_train == -1][:15000]
# Y_train_2 = Y_train[Y_train == -2][:8000]
# Y_train_3 = Y_train[Y_train == -3][:8000]
# Y_train_4 = Y_train[Y_train == -4][:8000]
# X_train = np.concatenate((X_train_normal, X_train_1, X_train_2, X_train_3, X_train_4), axis=0)
# Y_train = np.concatenate((Y_train_normal, Y_train_1, Y_train_2, Y_train_3, Y_train_4), axis=0)
# array_index = np.arange(len(Y_train))
# np.random.shuffle(array_index)
# X_train = X_train[array_index,:]
# Y_train = Y_train[array_index]

#Summerize data:
print("=====================LOCAL DATA SUMMARY=========================")
print(f"Number of samples {X_train.shape[0]}, Number of features: {X_train.shape[1]}")
print(f"Train: Number of normal {np.sum(Y_train == 1)}, Number of attack {np.sum(Y_train == -1)}")
print(f"Number of labels: {print(np.unique(labels_idx))}")
print(f"Number of : {len(Y_train[Y_train == -1])}")
print(f"Number of : {len(Y_train[Y_train == -2])}")
print(f"Number of : {len(Y_train[Y_train == -3])}")
print(f"Number of : {len(Y_train[Y_train == -4])}")
print("=================================================================")
#Prepare model
print(f">> Prepare model and trainer ....")
local_trainer = NewOnlineAdaboost()
local_trainer.build(n_labels = hyper.n_labels, n_features=hyper.n_features) 
    
print(">> Training local model ...")
local_trainer.fit(X_train, Y_train)

#Get trained model
strong_gmms = local_trainer.strong_gmms
alphas = local_trainer.alphas 
    
#Send model to global nodes
print(">> Saving model ...")
nodeid = int(args.nodeid)
model_dict = get_model_dict(nodeid, strong_gmms, alphas)
params = json.dumps(model_dict)

if os.path.exists(f"checkpoint/local/kdd/local_model{nodeid}.json"):
    with open(f"checkpoint/local/kdd/local_model{nodeid}.json", "w") as outfile:
        outfile.write(params)
else:
    with open(f"checkpoint/local/kdd/local_model{nodeid}.json", "x") as outfile:
        outfile.write(params)

    
