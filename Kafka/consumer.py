from kafka import KafkaConsumer
import msgpack
import json

#Create new topic
consumer = KafkaConsumer('my-topic',
                         group_id='my-group',
                         bootstrap_servers=['localhost:9092'])

#consume earliest available messages, don't commit offsets
KafkaConsumer(auto_offset_reset='earliest', enable_auto_commit=False)


#consume msgpack
KafkaConsumer(value_deserializer=msgpack.unpackb)

#consume json messages
KafkaConsumer(value_deserializer=lambda m: json.loads(m.decode('ascii')))

for msg in consumer:
    print("%s:%d:%d: key=%s value = %s" %(msg.topic, msg.partition,
                                          msg.offset, msg.key,
                                          msg.value))
    


#StopIteration if no message after 1sec
#KafkaConsumer(consumer_timeout_ms=1000)

#Subscribe to a regex topic pattern
# consumer = KafkaConsumer()
# consumer.subscribe(pattern='^awesome.*')

#Use multiple consumers in parallel w/0.9 kafka brokers
#typically you would run each on a different server / process / CPU
# consumer1 = KafkaConsumer('my-topic',
#                           group_id='my-group',
#                           bootstrap_servers='my.server.com')
# consumer2 = KafkaConsumer('my-topic',
#                           group_id='my-group',
#                           bootstrap_servers='my.server.com')