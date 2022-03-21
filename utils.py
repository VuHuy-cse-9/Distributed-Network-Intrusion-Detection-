import numpy as np
from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import msgpack
import json
from GMMs import hyper
from GMMs import gmms

#GLOBAL VARIABLES
model_producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda m: json.dumps(m).encode('ascii')
)

data_producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda m: json.dumps(m).encode('ascii')
)

model_consumer = KafkaConsumer('model-topic',
                        bootstrap_servers=['localhost:9092'],
                        value_deserializer=lambda m: json.loads(m.decode('ascii')),
                        auto_offset_reset="earliest")

data_consumer = KafkaConsumer('data-topic',
                        bootstrap_servers=['localhost:9092'],
                        value_deserializer=lambda m: json.loads(m.decode('ascii')))

#HELP FUNCTION
def detection_rate(labels, predicts):
    """Calculate detection rate (TPR)

    Args:
        labels (_array-like (N_test,)_): _label of test dataset_
        predicts (_array-like (N_test_): _label predicted by model_
    return:
        dtr (float): detection rate
    """
    if predicts.shape != labels.shape:
        raise Exception(f"Shape of predicts: {predicts.shape} is not equal shape of lables: {labels.shape}")
    return np.sum((labels - predicts)[labels == -1] == 0, dtype=np.float32) / np.sum(labels == -1)

def false_alarm_rate(labels, predicts):
    """Calculate false alarm rate
    Args:
        labels (_array-like (N_test,)_): _label of test dataset_
        predicts (_array-like (N_test_): _label predicted by model_
    return:
        far: (float): false alarm rate
    """
    if predicts.shape != labels.shape:
        raise Exception(f"Shape of predicts: {predicts.shape} is not equal shape of lables: {labels.shape}")
    return np.sum((labels - predicts)[labels == 1] != 0, dtype=np.float32) / np.sum(labels == 1)

def send_model(model_dict):
    future = model_producer.send('model-topic', model_dict)
    try:
        record_metadata = future.get(timeout=10)
    except KafkaError:
    #Decide what to do if produce request failed
        log.exception()
    pass
    
def send_data(X, y):
    data_producer.send(
        "data-topic", {
            'y': y,
            'X': X  #Fix BUG here
        }
    )
    
def _get_models():
    count_nodes = hyper.N_nodes
    node_models = []
    print(">> Begin to clone model")
    for msg in model_consumer:
        print(">> Get model")
        node_models.append(msg.value)
        count_nodes -= 1
        if count_nodes == 0:
            break
    return node_models

def _get_data():
    """Get data from local nodes
    """
    count_data = hyper.N_data_global
    X, y = [], []
    print(">> Begin to clone data")
    for msg in data_consumer:
        X.append(msg.value["X"])
        y.append(msg.value["y"])
        count_data -= 1
        if count_data == 0:
            break
    X, y = np.array(X), np.array(y)
    if X.shape[0] != hyper.N_data_global or X.shape[0] != y.shape[0]:
        raise Exception(f"Shape of X from _get_data, utils is not right: {X.shape}, or not equal y: {y.shape}")
    return X, y
    

def load_fake_data(N_train, N_test, N_attribute):
    #train dataset
    X0 = np.random.normal(loc=2.0, scale=1.0, size=(N_train, N_attribute))
    X1 = np.random.normal(loc=5.0, scale=1.0, size=(N_train, N_attribute))
    X_train = np.concatenate((X0, X1))
    Y_train = np.concatenate((np.ones((N_train, ), dtype=np.int32),
                        -1*np.ones((N_train, ), dtype=np.int32)))
    

    shuffle_train_index = np.arange(2*N_train)
    np.random.shuffle(shuffle_train_index)
    X_train  = X_train[shuffle_train_index]
    Y_train = Y_train[shuffle_train_index]

    

    #Test dataset
    X0 = np.random.normal(loc=2.0, scale=0.3, size=(N_test, N_attribute))
    X1 = np.random.normal(loc=4.0, scale=0.3, size=(N_test, N_attribute))
    X_test = np.concatenate((X0, X1), axis=0)
    Y_test = np.concatenate((np.ones((N_test, ), dtype=np.int32), 
                            -1*np.ones((N_test, ), dtype=np.int32)))
    
    shuffle_test_index = np.arange(2*N_test)
    np.random.shuffle(shuffle_test_index)
    X_test  = X_test[shuffle_test_index]
    Y_test = Y_test[shuffle_test_index]
    
    
    if X_train.shape != (2*N_train, N_attribute) or \
        Y_train.shape != (2*N_train, ):
            raise Exception(f"X_train, or Y_train has changed their shape: {X_train.shape}, {Y_train.shape}")
        
    if X_test.shape != (2*N_test, N_attribute) or \
        Y_test.shape != (2*N_test, ):
            raise Exception(f"X_test, or Y_test has changed their shape: {X_test.shape}, {Y_test.shape}")
        
    return X_train, X_test, Y_train, Y_test

def split_train_test(X, y, ratio):
    """Split dataset into train, testset 

    Args:
        X (_N_samples, N_features_): Samples_
        y (_N_samples_): Labels
        ratio: N_train / N_samples
    """
    if X.shape[0] != y.shape[0]:
        raise Exception(f"Number of sample in X not the same in y: {X.shape[0]}, {y.shape[0]}")
    
    if ratio >= 1.0:
        raise Exception(f"Ratio cannot larger than 1.0")
    
    N_samples = X.shape[0]
    N_train = int(ratio * N_samples)
    shuffle_index = np.arange(N_samples)
    np.random.shuffle(shuffle_index)
    train_index = shuffle_index[:N_train]
    test_index = shuffle_index[N_train:]
    
    X_train, y_train = np.squeeze(X[train_index]), np.squeeze(y[train_index])
    X_test, y_test = np.squeeze(X[test_index]), np.squeeze(y[test_index])
    
    if X_train.shape != (N_train, X.shape[1]) or y_train.shape != (N_train, ):
        raise Exception(f"Shape of X_train or y_train is not right, {X_train.shape}, {y_train.shape}")
    
    if X_test.shape != (N_samples - N_train, X.shape[1]) or y_test.shape != (N_samples - N_train, ):
        raise Exception(f"Shape of X_test or y_test is not right, {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def clone_model_from_local(curr_nodeid, N_nodes, N_classifiers):
    model_params = _get_models()
    local_models = np.empty((N_nodes, N_classifiers), dtype=gmms.OnlineGMM)
    alphas = np.empty((N_nodes, N_classifiers))
    for node_index in range(N_nodes):
        nodeid = int(model_params[node_index]["node"])
        if nodeid == curr_nodeid: continue
        alphas[nodeid] = np.array(model_params[node_index]["alphas"])
        for index in range(N_classifiers):
            local_models[nodeid, index] = gmms.OnlineGMM(hyper.std, n_components=hyper.n_components, T=1)
            local_models[nodeid, index].set_parameters(model_params[node_index][f"model_{index}"])
    print(">> Clone model from local: Done!")
    return local_models, alphas

def clone_data_from_local(ratio=0.8):
    X, y = _get_data()
    X_train, X_test, y_train, y_test = split_train_test(X, y, ratio=0.8)
    print(">> Clone data from local: Done!")
    return X_train, X_test, y_train, y_test


    
    