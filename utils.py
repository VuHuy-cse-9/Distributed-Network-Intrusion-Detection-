import numpy as np
from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import msgpack
import json

#GLOBAL VARIABLES
model_producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda m: json.dumps(m).encode('ascii')
)

data_producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda m: json.dumps(m).encode('ascii')
)

model_consumer = KafkaConsumer('model-local-global',
                        bootstrap_servers=['localhost:9092'],
                        value_deserializer=lambda m: json.loads(m.decode('ascii')))

#HELP FUNCTION
def detection_rate(labels, predicts):
    """Calculate detection rate (TPR)

    Args:
        labels (_array-like (N_test,)_): _label of test dataset_
        predicts (_array-like (N_test_): _label predicted by model_
    """
    # print(f"TP: {np.sum((labels - predicts)[labels == -1] == 0, dtype=np.float32)}")
    # print(f"TP + FN: {np.sum(labels == -1)}")
    # print(f"labels: {labels.shape}")
    # print(f"predicts: {predicts.shape}")
    return np.sum((labels - predicts)[labels == -1] == 0, dtype=np.float32) / np.sum(labels == -1)

def false_alarm_rate(labels, predicts):
    """Calculate false alarm rate
    Args:
        labels (_array-like (N_test,)_): _label of test dataset_
        predicts (_array-like (N_test_): _label predicted by model_
    """
    # print(f"FP: {np.sum((labels - predicts)[labels == 1] != 0, dtype=np.float32)}")
    # print(f"FP + TN: {np.sum(labels == 1)}")
    return np.sum((labels - predicts)[labels == 1] != 0, dtype=np.float32) / np.sum(labels == 1)

def send_model(model_dict):
    future = model_producer.send('model-local-global', model_dict)
    
    try:
        record_metadata = future.get(timeout=10)
    except KafkaError:
    #Decide what to do if produce request failed
        log.exception()
    pass
    
def send_data(X, y):
    data_producer.send(
        "data-local-global", {
            'y': y,
            'X': X  #Fix BUG here
        }
    )
    
def get_models():
    count = 1
    node_models = []
    for msg in model_consumer:
        print(">> Receive model")
        node_models.append(msg.value)
        #print(msg.value)
        count -= 1
        if count == 0:
            break
    return node_models
    

    