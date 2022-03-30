from models.GMM import OnlineGMM
from utils import detection_rate, false_alarm_rate
from DataGenerator.DataGenerator import split_train_test, load_fake_data
from visualize import plot_one_norm_data
import hyper

import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

#Fake data
N_train = 250000
N_test = 200
X_train, X_test, Y_train, Y_test = load_fake_data(N_train, N_test, 1)

model = OnlineGMM(std=1.0, n_components=4, T=0.5)
model.build(n_labels=2)

#Visualize:
print(">> Visualize before training ...")
means = model.means[None, :, :] 
stds = model.stds[None, :, :]
print(f">> means: {means.shape}")
print(f">> stds: {stds.shape}")
plot_one_norm_data(X_train[:, 0][:, None], Y_train, means, stds)

N_train = X_train.shape[0]

print(">> Training ...")
for x, y in tqdm(zip(X_train[:, 0][:, None], Y_train), total=N_train):
    model.fit(x, y)
    
model.predict(X_test)
    
#print(">> AFTER TRAIN")
#Visualize:
print(">> Visualize after training ...")
means = model.means[None, :, :]
stds = model.stds[None, :, :]
plot_one_norm_data(X_train[:, 0][:, None], Y_train, means, stds)

print("=====================================")

