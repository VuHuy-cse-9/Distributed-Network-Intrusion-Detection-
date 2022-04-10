import numpy as np
from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import msgpack
import json
import hyper
from models.GMM import OnlineGMM

#GLOBAL VARIABLES

def get_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=[hyper.boot_strap_server],
        value_serializer=lambda m: json.dumps(m).encode('ascii')
    )

# data_producer = KafkaProducer(
#     bootstrap_servers=["localhost:9092"],
#     value_serializer=lambda m: json.dumps(m).encode('ascii')
# )
def get_kafka_consumer():
    return KafkaConsumer('model-topic',
                            bootstrap_servers=[hyper.boot_strap_server],
                            value_deserializer=lambda m: json.loads(m.decode('ascii')),
                            auto_offset_reset="earliest")

# data_consumer = KafkaConsumer('data-topic',
#                         bootstrap_servers=['localhost:9092'],
#                         value_deserializer=lambda m: json.loads(m.decode('ascii')))

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
    return np.sum((labels == predicts)[labels == -1], dtype=np.float32) / np.sum(labels == -1)

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
    return np.sum((labels != predicts)[labels == 1], dtype=np.float32) / np.sum(labels == 1)

def send_model(model_dict):
    model_producer = get_kafka_producer()
    future = model_producer.send('model-topic', model_dict)
    try:
        record_metadata = future.get(timeout=10)
    except KafkaError:
    #Decide what to do if produce request failed
        log.exception()
    pass
    
# def send_data(X, y):
#     data_producer.send(
#         "data-topic", {
#             'y': y,
#             'X': X  #Fix BUG here
#         }
#     )
    
def _get_models():
    count_nodes = hyper.N_nodes
    node_models = []
    model_consumer = get_kafka_consumer()
    for msg in model_consumer:
        node_models.append(msg.value)
        count_nodes -= 1
        if count_nodes == 0:
            break
    return node_models

# def _get_data():
#     """Get data from local nodes
#     """
#     count_data = hyper.N_data_global
#     X, y = [], []
#     for msg in data_consumer:
#         X.append(msg.value["X"])
#         y.append(msg.value["y"])
#         count_data -= 1
#         if count_data == 0:
#             break
#     X, y = np.array(X), np.array(y)
#     if X.shape[0] != hyper.N_data_global or X.shape[0] != y.shape[0]:
#         raise Exception(f"Shape of X from _get_data, utils is not right: {X.shape}, or not equal y: {y.shape}")
#     return X, y

def convert_json_to_local_models(model_params, curr_nodeid, N_nodes, N_classifiers):
    local_models = np.empty((N_nodes, N_classifiers), dtype= OnlineGMM)
    alphas = np.empty((N_nodes, N_classifiers))
    for node_index in range(N_nodes):
        nodeid = int(model_params[node_index]["node"])
        if nodeid == curr_nodeid: continue
        alphas[nodeid] = np.array(model_params[node_index]["alphas"])
        for index in range(N_classifiers):
            local_models[nodeid, index] = OnlineGMM(hyper.std, n_components=hyper.n_components, T=hyper.T)
            local_models[nodeid, index].set_parameters(model_params[node_index][f"model_{index}"])
    return local_models, alphas

def clone_model_from_local(curr_nodeid, N_nodes, N_classifiers):
    model_params = _get_models()
    local_models, alphas = convert_json_to_local_models(model_params, curr_nodeid, N_nodes, N_classifiers)
    return local_models, alphas

# def clone_data_from_local(ratio=0.8):
#     X, y = _get_data()
#     X_train, X_test, y_train, y_test = split_train_test(X, y, ratio=0.8)
#     return X_train, X_test, y_train, y_test

def get_model_dict(nodeid, local_models, alphas):
    model_dict = {}
    model_dict["node"] = nodeid
    model_dict["alphas"] = alphas.tolist()
    for index, gmms in enumerate(local_models):
        model_dict[f"model_{index}"] = gmms.get_parameters()
    return model_dict

    
    