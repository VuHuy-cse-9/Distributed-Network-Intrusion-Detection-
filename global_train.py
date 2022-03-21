import numpy as np
from sklearn.svm import SVC
from utils import detection_rate, clone_data_from_local, clone_model_from_local, false_alarm_rate
from GMMs import gmms
from GMMs import hyper
from ThreadWithResult import ThreadWithResult
from visualize import plot_global_history

print(">> Initialize hyperameter ...")
#Hyperameter
Vmax = hyper.V_max
Q = hyper.N_states
N_nodes = hyper.N_nodes
tau = hyper.tau
N_classifiers = hyper.N_features
N_iter = hyper.N_iter
eta = 1e-5
w = hyper.w #inertia weight
c1, c2 = hyper.c1, hyper.c2
u1, u2 = hyper.u1, hyper.u2

print(">> Load parameters from local nodes ...")
#CLONE FROM LOCAL:
thread_model = ThreadWithResult(target=clone_model_from_local, args=(N_nodes, N_classifiers,))
thread_data = ThreadWithResult(target=clone_data_from_local, args=(0.8,))
thread_model.start()
thread_data.start()
thread_model.join()
thread_data.join()
local_models, alphas = thread_model.result              #Local models, shape = (N_nodes, N_features, )
X_train, X_test, Y_train, Y_test = thread_data.result   #Dataset, X = (N_samples, N_features), Y = (N_samples, )

if (np.sum(alphas, axis=1) > 1).any():
    raise Exception("Alphas total larger than 1")

print(f">> Alphas: {alphas.shape}")
print(f">> local models: {local_models.shape}")
print(f">> X_train: {X_train.shape}")
print(f">> X_test: {X_test.shape}")
print(f">> Y_train: {Y_train.shape}")
print(f">> Y_test: {Y_test.shape}")

print(">> Dataset summery:")
print(f"Number of data: {X_train.shape[0] + X_test.shape[0]}")
print(f"Number of normal data in train: {np.sum(Y_train == 1)}")
print(f"Number of normal data in test: {np.sum(Y_test == 1)}")
print(f"Number of attack data in train: {np.sum(Y_train < 1)}")
print(f"Number of attack data in test: {np.sum(Y_test < 1)}")

#VARIABLES:
Si = np.zeros((Q, N_nodes)) #Best state of particles, equation (35)
Si_fit = None               #Fitness value of best state of particles
Sg = np.zeros([N_nodes,])             #Best global state of every particles
Vi = np.zeros((Q, N_nodes)) #Velocities of each particles, equation (37)
history = {
    "Si_fit": [],   #Best fit of particle, (N_iter, N_states)
    "Sg_fit": [],   #Global fit, (N_iter, )
    "L": [],        #Particle state, (N_iter, N_states)
    "DETR": [],     #Detection rate (N_iter, N_states)
    "FAR": []       #False alarm rate (N_iter, N_states)
}
#INITIALIZE MODELS:
svcs = []
for Q_index in range(Q):
    svcs.append(SVC(kernel = 'rbf', C = 1e5)) #Every states have a private support vector machine

#HELP FUNCTION
def data_to_vector(Data):
    """ Transform raw data into feature vector using Local models
    Args:
        Data (array_(N_samples, N_features)): dataset
    Returns:
        r: (array (N_samples, N_nodes)), feature vector
    """
    r = np.zeros((N_nodes, Data.shape[0]))
    for node_index in np.arange(N_nodes): 
        r[node_index] = \
            np.sum([alphas[node_index, i] * \
                local_models[node_index, i].predict(Data[:, i]) \
                    for i in range(N_classifiers)], axis=0)
        print(f">> alphas: {alphas[node_index, 0]}, {alphas[node_index][0]}")
        
    #alphas[inode][n_classifier] * 
    return np.transpose(r, (1, 0))

def fitness_function(dtr, tau, L):
    """Calculate fitness value using equation (34)

    Args:
        dtr (array (N_states, )): detection rate by private svm on test dataset.
        tau (_hyperameter_): _hyperameter, equation (34)
        L (array (N_states,): Number of nodes be selected on each state, equation (34)

    Returns:
        : (array, (N_states,)): fitness value for each state
    """
    #BUG: IF NODE IS CHOOSE, LOG ALWAY NEGATIVE, ESPECIALLY ONE NODE?
    return tau * dtr + (1 - tau) * np.log(np.maximum((N_nodes - L) / (N_nodes * 1.0), eta))

def constraint_velocity(V):
    """ Limit the range of value of V
    Args:
        V (_Q, Nodes_): _Velocities for eachs states, equation (38)
    """
    #TODO: Add constrainted
    return V
    
#TRAIN
print(">> Training ....")
#Initialize state
#BUG: TWO Q MAY HAVE SAME STATES!
X_state = np.random.normal(loc=0.25, scale=0.5, size=(Q, N_nodes))
X_state = X_state  * (X_state > 0)
Si = X_state  


for step in range(N_iter):
    print(f"=============STEP {step + 1}===============")
   #MODIFY: L(int) to L(weight) in (34)
    L_weight = np.sum(X_state, axis=1)

    history["L"].append(L_weight)
    
    if L_weight.shape != (Q,):
        raise Exception(f"Expect L_weight has shape {(Q, )}, instead {L_weight.shape}") 
    
    #Training SVM
    r_train = data_to_vector(X_train)
    if r_train.shape != (X_train.shape[0], N_nodes):
        raise Exception(f"Expect r_train has shape {(X_train.shape[0], N_nodes)}, instead {r_train.shape}")   
    print(f">> r_train: {r_train.shape}")
    if (r_train > 1.0).any() or (r_train < -1.0).any():
        raise Exception("r_train exist value > 1.0 or <-1.0")
    r_train = X_state[:, None, :] * r_train[None, :, :]
    
    for Q_index in range(Q):
        svcs[Q_index].fit(r_train[Q_index], Y_train)    
    
    
    #Evaluate svm:
    r_test = data_to_vector(X_test)
    if r_test.shape != (X_test.shape[0], N_nodes):
        raise Exception(f"Expect r_test has shape {(X_test.shape[0], N_nodes)}, instead {r_test.shape}")    
    print(f">> r_test: {r_test.shape}")
    r_test = X_state[:, None, :] * r_test[None, :, :]
    
    dtr = np.zeros((Q, ))  #Detection rate
    far = np.zeros((Q, ))  #False alarm rate
    for Q_index in range(Q):
        predicts = svcs[Q_index].predict(r_test[Q_index])
        dtr[Q_index] =  detection_rate( 
                            Y_test, 
                            predicts
                        )
        far[Q_index] = false_alarm_rate(
                            Y_test, 
                            predicts
                        )       
    #MODIFY: STATE HAVE L = 0 -> DET = -10.0
    dtr[L_weight == 0] = 0.0
    history["DETR"].append(dtr)
    history["FAR"].append(far)
    #Calculate fitness & Update best state
    X_fit = fitness_function(dtr, tau=tau, L=L_weight)
    if X_fit.shape != (Q,):
        raise Exception(f"Expect X_fit has shape {(Q,)}, instead {X_fit.shape}")
    
    if Si_fit is None:
        Si_fit = X_fit
    else:
        for Q_index in range(Q):
            if X_fit[Q_index] > Si_fit[Q_index]:
                Si[Q_index] = X_state[Q_index]
                Si_fit[Q_index] = X_fit[Q_index]
    if Si.shape != (Q,N_nodes):
        raise Exception(f"Expect Si has shape {(Q, N_nodes)}, instead {Si.shape}")
                
    #Update global state:
    Sg = Si[np.argmax(Si_fit)]
    Sg_fit = Si_fit[np.argmax(Si_fit)]
    if Sg.shape != (N_nodes,):
        raise Exception(f"Expect Sg has shape {(N_nodes,)}, instead {Sg.shape}")
    
    history["Si_fit"].append(Si_fit)
    history["Sg_fit"].append(Sg_fit)
    
    for Q_index in range(Q):
        #Calculate velocities
        Vi[Q_index] = constraint_velocity(
            w * Vi[Q_index] + c1 * u1 * (Si[Q_index] - X_state[Q_index]) + c2 * u2 * (Sg - X_state[Q_index])
        )
        #Evolve particle state
        X_state[Q_index] += Vi[Q_index]
    #print(f">> r_train: {r_train}, r_test: {r_test}")
        
    for Q_index in range(Q):
        print(f">> Q: {Q_index}, STATE: {X_state[Q_index]}, Vi: {Vi[Q_index]}, DTR: {dtr[Q_index]}, FAR: {far[Q_index]}, FIT: {X_fit[Q_index]}, BFIT: {Si_fit[Q_index]}")
    print(f">> GSTATE: {Sg}, GFIT: {Sg_fit}")
    
plot_global_history(history=history, N_iter=N_iter, N_states=Q)