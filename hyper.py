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
