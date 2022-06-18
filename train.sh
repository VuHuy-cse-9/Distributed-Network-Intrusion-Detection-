#ip 22:
#python3 train.py --nodeid 0 --train_mode full --gamma 10 --log_transform True --path_save_dir checkpoint/local
#python3 train.py --nodeid 4 --train_mode full --gamma 15 --log_transform True --path_save_dir checkpoint/local
#python3 train.py --nodeid 5 --train_mode full --gamma 15 --log_transform True --path_save_dir checkpoint/local

#ip 7:
#python3 train.py --nodeid 0 --cate2cont remove --train_mode full --gamma 10 --log_transform True --path_save_dir checkpoint/local
#python3 train.py --nodeid 1 --cate2cont remove --train_mode full --gamma 10 --log_transform True --path_save_dir checkpoint/local

#ip 4:
#python3 train.py --nodeid 2 --train_mode full --gamma 5 --log_transform True --path_save_dir checkpoint/local
#python3 train.py --nodeid 3 --train_mode full --gamma 5 --log_transform True --path_save_dir checkpoint/local

#
# GLOBAL TRAINING SCRIPT: 
#

#Ip 7:
#python3 train.py --nodeid 0 --islocal False --N_nodes 6 --inertia_weight_mode fix --kafka_server 10.1.6.7:9092
#python3 train.py --nodeid 1 --islocal False --N_nodes 6 --inertia_weight_mode fix --kafka_server 10.1.6.7:9092

#Ip 4:
#python3 train.py --nodeid 2 --islocal False --N_nodes 6 --inertia_weight_mode fix --kafka_server 10.1.6.7:9092
#python3 train.py --nodeid 3 --islocal False --N_nodes 6 --inertia_weight_mode fix --kafka_server 10.1.6.7:9092

#Ip 22:
#python3 train.py --nodeid 4 --islocal False --N_nodes 6 --inertia_weight_mode fix --kafka_server 10.1.6.7:9092
#python3 train.py --nodeid 5 --islocal False --N_nodes 6 --inertia_weight_mode fix --kafka_server 10.1.6.7:9092