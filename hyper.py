#Local
n_features = 41
n_labels = 5
#Global
# n_features = 41
# n_labels = 5
#Random
# n_features = 41
# n_labels = 2
#GMMS HYPERAMETER
gamma = 10   #Equation (25)
r = 0.5     #Equation (32)
p = 0.1     #Equation (23)
beta = 0.8  #Equation (29)  
std = 1.0   #Equation (15)
n_components = 4
T = 0.5
#LOCAL HYPERAMTER
P = 8      #predefined maximum number of iterations
            #Equation (25)
#GLOBAL HYPERAMETER
N_states = 3
N_nodes = 6
N_data_local_send = 500
N_data_global = N_data_local_send * N_nodes 
tau = 0.75 #Equation (34), weight
u1, u2 = 0.2, 0.3 #Independent random value, equation (37)
c1, c2 = 0.1, 0.1 #Acceleration constants, equation (37)
w = 1.4 #inertia weight
N_iter = 20
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


skew_features = [
    "dst_host_count",
    "dst_host_srv_count",
    "srv_count",
    "count",
    "num_root",
    "num_compromised",
    "dst_bytes",
    "src_bytes",
    "duration"
]



path_train = "dataset/kddcup1999_10percent.csv"
path_test = "dataset/test.csv"

dos = ["back.", "land.", "neptune.", "pod.", "smurf.", "teardrop."]
u2r = ["buffer_overflow.", "loadmodule.", "perl.", "rootkit."]
r2l = ["ftp_write.", "guess_passwd.", "imap.", "multihop.", "phf.", "spy.", "warezclient.", "warezmaster."]
prob = ["ipsweep.", "nmap.", "portsweep.", "satan."]


attack_global_name = ["normal.", "smurf.", "neptune.", "snmpgetattack.", "mailbomb."]

attack_global_train = {
    "normal.": 1,
    "smurf.": -1,
    "neptune.": -2,
    "snmpgetattack.": -3,
    "mailbomb.": -4
}

limit = [
    {
        "neptune.":5000,
        "smurf.": 5000,
        "snmpgetattack.":0,
        "mailbomb.":0,
        "normal.":10000
    },
    {
        "neptune.":1000,
        "smurf.": 1000,
        "snmpgetattack.":7741,
        "mailbomb.":0,
        "normal.":10000
    },
    {
        "neptune.":1000,
        "smurf.": 1000,
        "snmpgetattack.":0,
        "mailbomb.":5000,
        "normal.":10000
    },
    {
        "neptune.":3500,
        "smurf.": 3500,
        "snmpgetattack.":1500,
        "mailbomb.":1500,
        "normal.":10000
    },
    {
        "neptune.":2500,
        "smurf.": 2500,
        "snmpgetattack.":2500,
        "mailbomb.":2500,
        "normal.":10000
    },
    {
        "neptune.":1500,
        "smurf.": 1500,
        "snmpgetattack.":3500,
        "mailbomb.":3500,
        "normal.":10000
    },
]

boot_strap_server = "10.1.6.7:9092"
