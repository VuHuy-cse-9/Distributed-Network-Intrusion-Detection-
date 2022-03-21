"""Training GMMs using online Adaboost
"""
import numpy as np
from GMMs import gmms
from GMMs import hyper
from utils import detection_rate, false_alarm_rate, send_model, load_fake_data, send_data, clone_model_from_local
import json
from tqdm import tqdm
from visualize import plot_multi_norm_data, plot_global_history
import argparse
from sklearn.svm import SVC
from local_train import NewOnlineAdaboost
from global_train import PSOSVMTrainer

#ARGUMENTS:
parser = argparse.ArgumentParser(description='Distributed Intrusion Detection')
parser.add_argument('--nodeid', default=1.0, type=int, help='Node"s id')
args = parser.parse_args()

#Hyper:
N_classifiers = hyper.N_features

#Load data:
N_labels, N_train, N_test = 2, 500, 100
print(">> Loading dataset ...")
X_train, X_test, Y_train, Y_test = load_fake_data(N_train, N_test, N_classifiers)
N_train = N_train * 2 #JUST FOR TEMPORARY
N_test = N_test * 2

#Summerize data:
print("=====================LOCAL DATA SUMMARY=========================")
print(f"Train: Number of normal {np.sum(Y_train == 1)}, Number of attack {np.sum(Y_train == -1)}")
print(f"Test: Number of normal {np.sum(Y_test == 1)}, Number of attack {np.sum(Y_test == -1)}")
    
#Prepare model
print(f">> Prepare model and trainer ....")
local_trainer = NewOnlineAdaboost()
local_trainer.build() 
    
#Visualize:
print(">> Visualize before training ...")
gmms = local_trainer.gmms
means = np.array([gmms[i].means for i in range(N_classifiers)])
stds = np.array([gmms[i].stds for i in range(N_classifiers)])
plot_multi_norm_data(X_train, Y_train, means, stds)

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
print(">> Visualize after training ...")
means = np.array([strong_gmms[i].means for i in range(N_classifiers)])
stds = np.array([strong_gmms[i].stds for i in range(N_classifiers)])
plot_multi_norm_data(X_train, Y_train, means, stds)
    
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

# for step in range(N_iter):
#     print(f"=============STEP {step + 1}===============")
#    #MODIFY: L(int) to L(weight) in (34)
#     L_weight = np.sum(X_state, axis=1)

#     history["L"].append(L_weight)
    
#     if L_weight.shape != (Q,):
#         raise Exception(f"Expect L_weight has shape {(Q, )}, instead {L_weight.shape}") 
    
#     #Training SVM
#     r_train = data_to_vector(X_global_train)
#     if r_train.shape != (X_global_train.shape[0], N_nodes):
#         raise Exception(f"Expect r_train has shape {(X_global_train.shape[0], N_nodes)}, instead {r_train.shape}")   
#     print(f">> r_train: {r_train.shape}")
#     if (r_train > 1.0).any() or (r_train < -1.0).any():
#         raise Exception("r_train exist value > 1.0 or <-1.0")
#     r_train = X_state[:, None, :] * r_train[None, :, :]
    
#     for Q_index in range(Q):
#         svcs[Q_index].fit(r_train[Q_index], Y_global_label)    
    
    
#     #Evaluate svm:
#     r_test = data_to_vector(X_test)
#     if r_test.shape != (X_test.shape[0], N_nodes):
#         raise Exception(f"Expect r_test has shape {(X_test.shape[0], N_nodes)}, instead {r_test.shape}")    
#     print(f">> r_test: {r_test.shape}")
#     r_test = X_state[:, None, :] * r_test[None, :, :]
    
#     dtr = np.zeros((Q, ))  #Detection rate
#     far = np.zeros((Q, ))  #False alarm rate
#     for Q_index in range(Q):
#         predicts = svcs[Q_index].predict(r_test[Q_index])
#         dtr[Q_index] =  detection_rate( 
#                             Y_test, 
#                             predicts
#                         )
#         far[Q_index] = false_alarm_rate(
#                             Y_test, 
#                             predicts
#                         )       
#     #MODIFY: STATE HAVE L = 0 -> DET = -10.0
#     dtr[L_weight == 0] = 0.0
#     history["DETR"].append(dtr)
#     history["FAR"].append(far)
#     #Calculate fitness & Update best state
#     X_fit = fitness_function(dtr, tau=tau, L=L_weight)
#     if X_fit.shape != (Q,):
#         raise Exception(f"Expect X_fit has shape {(Q,)}, instead {X_fit.shape}")
    
#     if Si_fit is None:
#         Si_fit = X_fit
#     else:
#         for Q_index in range(Q):
#             if X_fit[Q_index] > Si_fit[Q_index]:
#                 Si[Q_index] = X_state[Q_index]
#                 Si_fit[Q_index] = X_fit[Q_index]
#     if Si.shape != (Q,N_nodes):
#         raise Exception(f"Expect Si has shape {(Q, N_nodes)}, instead {Si.shape}")
                
#     #Update global state:
#     Sg = Si[np.argmax(Si_fit)]
#     Sg_fit = Si_fit[np.argmax(Si_fit)]
#     if Sg.shape != (N_nodes,):
#         raise Exception(f"Expect Sg has shape {(N_nodes,)}, instead {Sg.shape}")
    
#     history["Si_fit"].append(Si_fit)
#     history["Sg_fit"].append(Sg_fit)
    
#     for Q_index in range(Q):
#         #Calculate velocities
#         Vi[Q_index] = constraint_velocity(
#             w * Vi[Q_index] + c1 * u1 * (Si[Q_index] - X_state[Q_index]) + c2 * u2 * (Sg - X_state[Q_index])
#         )
#         #Evolve particle state
#         X_state[Q_index] += Vi[Q_index]
#     #print(f">> r_train: {r_train}, r_test: {r_test}")
        
    # for Q_index in range(Q):
    #     print(f">> Q: {Q_index}, STATE: {X_state[Q_index]}, Vi: {Vi[Q_index]}, DTR: {dtr[Q_index]}, FAR: {far[Q_index]}, FIT: {X_fit[Q_index]}, BFIT: {Si_fit[Q_index]}")
    # print(f">> GSTATE: {Sg}, GFIT: {Sg_fit}")
    
