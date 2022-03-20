from kafka import KafkaConsumer
import msgpack
import json

#Create new topic
consumer = KafkaConsumer('model-topic',
                        bootstrap_servers=['localhost:9092'],
                        value_deserializer=lambda m: json.loads(m.decode('ascii')))

#consume earliest available messages, don't commit offsets
KafkaConsumer(auto_offset_reset='earliest', enable_auto_commit=False)


for msg in consumer:
    print("%s:%d:%d: key=%s value = %s" %(msg.topic, msg.partition,
                                          msg.offset, msg.key,
                                          msg.value))
    


#StopIteration if no message after 1sec
#KafkaConsumer(consumer_timeout_ms=1000)