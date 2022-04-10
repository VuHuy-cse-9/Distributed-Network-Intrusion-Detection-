from models.NewOnlineAdaboost import NewOnlineAdaboost
from models.PSOSVM import PSOSVMTrainer
from DataGenerator.DataGenerator import load_fake_data
import hyper
from utils import detection_rate, false_alarm_rate, send_model, clone_model_from_local
from visualize import plot_multi_norm_data, plot_global_history, plot_one_data_multi_norm, roc_curve_plot

import json
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.svm import SVC
from utils import get_model_dict
import os.path


#ARGUMENTS:
parser = argparse.ArgumentParser(description='Distributed Intrusion Detection')
parser.add_argument('--nodeid', default=1.0, type=int, help='Node"s id')
args = parser.parse_args()

#Hyper:
N_classifiers = hyper.n_features
#Load fake data:
N_labels, N_train, N_test = 2, 5000, 200
print(">> Loading dataset ...")
X_train, X_test, Y_train, Y_test = load_fake_data(N_train, N_test, N_classifiers)
N_train = N_train * 2 #JUST FOR TEMPORARY
N_test = N_test * 2

#Summerize data:
print("=====================LOCAL DATA SUMMARY=========================")
print(f"Number of samples {X_train.shape[0]}, Number of features: {X_train.shape[1]}")
print(f"Train: Number of normal {np.sum(Y_train == 1)}, Number of attack {np.sum(Y_train == -1)}")
print("=================================================================")
#Prepare model
print(f">> Prepare model and trainer ....")
local_trainer = NewOnlineAdaboost()
local_trainer.build(n_labels = hyper.n_labels, n_features=hyper.n_features) 
    
#Visualize:
print(">> Visualize before training ...")
# gmms = local_trainer.gmms
# means = np.array([gmms[i].means for i in range(N_classifiers)])
# stds = np.array([gmms[i].stds for i in range(N_classifiers)])
# plot_one_data_multi_norm(None, None, means, stds)

print(">> Training local model ...")
local_trainer.fit(X_train, Y_train)

#Get trained model
strong_gmms = local_trainer.strong_gmms
alphas = local_trainer.alphas 
    
#Visualize:
print(">> Visualize after training ...")
# means = np.array([strong_gmms[i].means for i in range(N_classifiers)])
# stds = np.array([strong_gmms[i].stds for i in range(N_classifiers)])
# plot_multi_norm_data(X_train, Y_train, means, stds)
    
# dtrs, fars = local_trainer.evaluate(X_test, Y_test)
# print(f">> detection rate: {dtrs[0]}, false alarm rate: {fars[0]}")

# y_score = local_trainer.pred_score(X_test)

# roc_curve_plot(Y_test, y_score)

print(">> Saving model ...")
nodeid = int(args.nodeid)
model_dict = get_model_dict(nodeid, strong_gmms, alphas)
params = json.dumps(model_dict)

if os.path.exists(f"checkpoint/local/rand/local_model{nodeid}.json"):
    with open(f"checkpoint/local/rand/local_model{nodeid}.json", "w") as outfile:
        outfile.write(params)
else:
    with open(f"checkpoint/local/rand/local_model{nodeid}.json", "x") as outfile:
        outfile.write(params)
    



    
