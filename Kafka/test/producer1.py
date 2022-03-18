from kafka import KafkaProducer
from kafka.errors import KafkaError
import msgpack
import json

producer = KafkaProducer(bootstrap_servers=["localhost:9092"], 
                        value_serializer=lambda m: json.dumps(m).encode('ascii'))

#Block for 'synchronous sends'
# try:
#     record_metadata = future.get(timeout=10)
# except KafkaError:
#     #Decide what to do if produce request failed
#     log.exception()
#     pass


#Produce asynchronously
for _ in range(100):
    producer.send('my-topic', {'key': ['value1', 'value2', 'value3']})
    
def on_send_success(record_metadata):
    print(record_metadata.topic)
    print(record_metadata.partition)
    print(record_metadata.offset)
    
def on_send_error(excp):
    log.error('I am an errback', exc_info=excp)
    #handle exception
    
#produce asynchronously with callbacks
# producer.send('my-topic', b'raw_bytes').add_callback(on_send_success).add_errback(on_send_error)

#Block until all async messages are sent
producer.flush()

#configure multiple retries
producer = KafkaProducer(retries=5)