import numpy as np
import hyper
import json
import pickle
from models.GMM import OnlineGMM
from models.NewOnlineAdaboost import NewOnlineAdaboost
from models.PSOSVM import PSOSVMTrainer
from utils import convert_json_to_local_models, detection_rate, false_alarm_rate
from DataGenerator.DataGenerator import load_fake_data
from visualize import roc_curve_plot

#====================================================================================
print(">> Loading Adboost combination classifiers ...")
local_models = []
for nodeid in range(hyper.N_nodes):
    #Save model params
    with open(f"checkpoint/global/rand/model{nodeid}.json", "rb") as file:
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
with open("checkpoint/global/rand/svm.pkl", "rb") as file:
    svm = pickle.load(file)
    
print(">> Loading Particle state ...")
state = None
with open("checkpoint/global/rand/state.json", "rb")  as file:
    state_param = json.load(file)
    state = state_param["state"]
    
global_model = PSOSVMTrainer()
global_model.set_params(svm, state, local_models)
    
print(">> Loading test dataset")
#====================================================================================
N_labels, N_train, N_test = 2, 2000, 500
_, X_test, _, Y_test = load_fake_data(N_train, N_test, hyper.n_features)
N_test = N_test * 2
print("===========DATA SUMMARY==============")
print(f"Number of sample: {X_test.shape[0]}")
print(f"Number of normal: {np.sum(Y_test == 1)}, number of attack: {np.sum(Y_test <= -1)}")
    
#Test global models:
y_pred = global_model.predict(X_test)

print(f"======PREDICT RESULT SUMMARY========")
print(f"shape of result: {np.array(y_pred).shape}")
print(f"Label that predict: {np.unique(y_pred)}")
print(f"Number that match: {np.sum(y_pred == Y_test)}")

print("========RESULT============")
dtr = detection_rate(Y_test, y_pred)
far = false_alarm_rate(Y_test, y_pred)
print(f"dtr: {dtr * 100}%, far: {far * 100}")

print(">> Global model: ROC curve")
roc_curve_plot(y_test=Y_test, y_score = y_pred)