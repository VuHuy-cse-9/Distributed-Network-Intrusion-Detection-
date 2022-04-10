import numpy as np
import hyper
import json
import pickle
from models.GMM import OnlineGMM
from models.NewOnlineAdaboost import NewOnlineAdaboost
from models.PSOSVM import PSOSVMTrainer
from utils import convert_json_to_local_models, detection_rate, false_alarm_rate
from DataGenerator.DataGenerator import get_data
from visualize import roc_curve_plot
import argparse

parser = argparse.ArgumentParser(description='Distributed Intrusion Detection')
parser.add_argument('--nodeid', default=1.0, type=int, help='Node"s id')
parser.add_argument('--option', default=1, type=int, help='Option for data preprocessing, 0 for category-continous converting, 1 for removing category features.')
args = parser.parse_args()

#====================================================================================
print(">> Loading Adboost combination classifiers ...")
local_models = []
for nodeid in range(hyper.N_nodes):
    #Save model params
    with open(f"checkpoint/global/kdd/model{nodeid}.json", "rb") as file:
        model_params = json.load(file)
        alphas = model_params["alphas"]
        strong_gmms_list = []
        for featureid in range(hyper.n_features):
            strong_gmms = OnlineGMM(1.0)
            strong_gmms.set_parameters(model_params[f"model_{featureid}"])
            strong_gmms_list.append(strong_gmms)
        strong_gmms_list = np.array(strong_gmms_list)
        local_model = NewOnlineAdaboost()
        local_model.set_params(strong_gmms_list, alphas, hyper.n_features)
        local_models.append(local_model)
local_models = np.array(local_models)

        
print(">> Loading SVM ...")
with open("checkpoint/global/kdd/svm.pkl", "rb") as file:
    svm = pickle.load(file)
    
print(">> Loading Particle state ...")
state = None
with open("checkpoint/global/kdd/state.json", "rb")  as file:
    state_param = json.load(file)
    state = state_param["state"]
    
global_model = PSOSVMTrainer()
global_model.set_params(svm, state, local_models)
    
#====================================================================================
print(">> Loading test dataset")
#====================================================================================
X_test, Y_test, labels_idx, labels, feature_names = get_data(hyper.path_test, hyper.category_features, hyper.skew_features, int(args.option))

print("=====================TEST DATA SUMMARY=========================")
print(f"Number of samples {X_test.shape[0]}, Number of features: {X_test.shape[1]}")
print(f"Train: Number of normal {np.sum(Y_test == 1)}, Number of attack {np.sum(Y_test <= -1)}")
print(f"Number of labels: {np.unique(Y_test).shape[0]}")
print(f"Number of : {len(Y_test[Y_test == -1])}")
print(f"Number of : {len(Y_test[Y_test == -2])}")
print(f"Number of : {len(Y_test[Y_test == -3])}")
print(f"Number of : {len(Y_test[Y_test == -4])}")
print(f"Number of : {len(Y_test[Y_test == -5])}")
print("=================================================================")
    
#Test global models:
Y_roc_test = np.array(np.clip(Y_test, -1, 1), dtype=np.float32)
y_pred = global_model.predict(X_test)

print(f"======PREDICT RESULT SUMMARY========")
print(f"shape of result: {np.array(y_pred).shape}")
print(f"Label that predict: {np.unique(y_pred)}")
print(f"Number that match: {np.sum(y_pred == Y_roc_test)}")

print("========RESULT============")
dtr = detection_rate(Y_roc_test, y_pred)
far = false_alarm_rate(Y_roc_test, y_pred)
print(f"dtr: {dtr * 100}%, far: {far * 100}")

print(">> Global model: ROC curve")
roc_curve_plot(y_test=Y_roc_test, y_score = y_pred)