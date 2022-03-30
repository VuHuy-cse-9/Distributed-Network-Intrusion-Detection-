#Data:
# n_features = 32
# n_labels = 6
n_features = 6
n_labels = 2
#GMMS HYPERAMETER
gamma = 25   #Equation (25)
r = 0.3     #Equation (32)
p = 0.1     #Equation (23)
beta = 0.8  #Equation (29)  
std = 1.0   #Equation (15)
n_components = 2
#LOCAL HYPERAMTER
P = 3      #predefined maximum number of iterations
            #Equation (25)
#GLOBAL HYPERAMETER
N_states = 3
N_nodes = 2
N_data_local_send = 500
N_data_global = N_data_local_send * N_nodes 
tau = 0.25 #Equation (34), weight
u1, u2 = 0.2, 0.3 #Independent random value, equation (37)
c1, c2 = 0.1, 0.1 #Acceleration constants, equation (37)
w = 0.2 #inertia weight
N_iter = 50
V_max = 2.0 #Equation (38)
category_features = [
    "protocol_type",
    "service",
    "flag",
    "land",
    "logged_in",
    "is_host_login",
    "is_guest_login",
    "num_access_files",
    "num_failed_logins",
]

path_train = "dataset/kddcup1999_10percent.csv"
path_test = "dataset/test.csv"

dos = ["back.", "land.", "neptune.", "pod.", "smurf.", "teardrop."]
u2r = ["buffer_overflow.", "loadmodule.", "perl.", "rootkit."]
r2l = ["ftp_write.", "guess_passwd.", "imap.", "multihop.", "phf.", "spy.", "warezclient.", "warezmaster."]
prob = ["ipsweep.", "nmap.", "portsweep.", "satan."]
