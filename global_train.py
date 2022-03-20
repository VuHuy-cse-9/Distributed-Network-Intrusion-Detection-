import numpy as np
from sklearn.svm import SVC
from utils import detection_rate, clone_data_from_local, clone_model_from_local, false_alarm_rate
from GMMs import gmms
from GMMs import hyper
from ThreadWithResult import ThreadWithResult

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

print(">> Dataset summery:")
print(f"Number of data: {X_train.shape[0] + X_test.shape[0]}")
print(f"Number of normal data in train: {np.sum(Y_train == 1)}")
print(f"Number of normal data in test: {np.sum(Y_test == 1)}")
print(f"Number of attack data in train: {np.sum(Y_train < 1)}")
print(f"Number of attack data in test: {np.sum(Y_test < 1)}")

#VARIABLES:
Si = np.zeros((Q, N_nodes)) #Best state of particles, equation (35)
Si_fit = None               #Fitness value of best state of particles
Sg = [N_nodes,]             #Best global state of every particles
Vi = np.zeros((Q, N_nodes)) #Velocities of each particles, equation (37)

#INITIALIZE MODELS:
svcs = []
for Q_index in range(Q):
    svcs.append(SVC(kernel = 'linear', C = 1e5)) #Every states have a private support vector machine

#HELP FUNCTION
def data_to_vector(Data, X_state):
    """ Transform raw data into feature vector using Local models
    Args:
        Data (array_(N_samples, N_features)): dataset
        X_state (array_(N_states, N_nodes)): states, give weight for each Node's models

    Returns:
        r: (array (N_states, N_samples, N_nodes)), feature vector
    """
    r = np.zeros((Q, N_nodes, Data.shape[0]))
    for Q_index in range(Q):
        for node_index in np.nonzero(X_state[Q_index]>0)[0]:    
            r[Q_index][node_index] = \
                X_state[Q_index, node_index] * \
                np.sum([alphas[node_index][i] * \
                    local_models[node_index, i].predict(Data[i]) \
                        for i in range(N_classifiers)])
    return np.transpose(r, (0, 2, 1))

def fitness_function(dtr, tau, L):
    """Calculate fitness value using equation (34)

    Args:
        dtr (array (N_states, )): detection rate by private svm on test dataset.
        tau (_hyperameter_): _hyperameter, equation (34)
        L (array (N_states,): Number of nodes be selected on each state, equation (34)

    Returns:
        : (array, (N_states,)): fitness value for each state
    """
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
    #Get number of nodes be used in particle i
    L = np.sum(X_state > 0, axis=1)
    
    #Training SVM
    r_train = data_to_vector(X_train, X_state)
    for Q_index in range(Q):
        svcs[Q_index].fit(r_train[Q_index], Y_train)
    
    if r_train.shape != (Q, X_train.shape[0], N_nodes):
        raise Exception(f"Expect r_train has shape {(Q, X_train.shape[0], N_nodes)}, instead {r_train.shape}")   
        
    #Evaluate svm:
    r_test = data_to_vector(X_test, X_state)
    if r_test.shape != (Q,X_test.shape[0], N_nodes):
        raise Exception(f"Expect r_test has shape {(Q,X_test.shape[0], N_nodes)}, instead {r_test.shape}")    
    
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
    
    #Calculate fitness & Update best state
    X_fit = fitness_function(dtr, tau=tau, L=L)
    if X_fit.shape != (Q,):
        raise Exception(f"Expect X_fit has shape {(Q,)}, instead {X_fit.shape}")
    
    if Si_fit is None:
        Si_fit = X_fit
    else:
        for Q_index in range(Q):
            if X_fit[Q_index] > Si_fit[Q_index]:
                Si[Q_index] = X_state[Q_index]
                Si_fit[Q_index] = X_state[Q_index]
    if Si.shape != (Q,N_nodes):
        raise Exception(f"Expect Si has shape {(Q, N_nodes)}, instead {Si.shape}")
                
    #Update global state:
    Sg = Si[np.argmax(Si_fit)]
    if Sg.shape != (N_nodes,):
        raise Exception(f"Expect Sg has shape {(N_nodes,)}, instead {Sg.shape}")
    
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
    print(f">> gFIT: {Sg}")