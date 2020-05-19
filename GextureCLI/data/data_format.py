import os 
import numpy as np
import math
import random
from pathlib import Path
import matplotlib.pyplot as plt

PATH = Path(os.path.dirname(os.path.realpath(__file__)))

def custom_append(data, sample):
    if data.size > 0:
        return np.append(data, sample, axis=0)
    else:
        return sample

def generate_sample(function, sample_class, sample_id):
    data_tmp = np.zeros((200, 3))
    for i in range(10):
        data_tmp = np.array([
        [
            sample_id,
            sample_class,
            function(t)
        ] for t in range(200)])
    return data_tmp

def generate_train(output_file, nb_samples):
    data = np.zeros(0)
    for i in range(nb_samples):
        data = custom_append(data, generate_sample(lambda x: math.cos(x/120) + random.random()/20, 1, i))
        data = custom_append(data, generate_sample(lambda x: math.sqrt(x)/math.sqrt(200) + random.random()/20, 2, i+10))
        data = custom_append(data, generate_sample(lambda x: math.exp(x/100)/math.exp(2) + random.random()/20, 3, i+20))
    np.savetxt(str(PATH / output_file), data, fmt='%s', delimiter=",")

def generate_input():
    input_sample = generate_sample(lambda x: math.cos(x/120) + random.random()/20,1, 1)
    input_sample = input_sample[:, 2:]
    np.savetxt(str(PATH / "input_data.csv"), input_sample, fmt='%s', delimiter=",")

def plot_gestures():
    plt.figure(1)
    plt.plot([math.cos(x/120) for x in range(200)])
    plt.plot([math.sqrt(x)/math.sqrt(200) for x in range(200)])
    plt.plot([math.exp(x/100)/math.exp(2) for x in range(200)])
    plt.show()

if __name__ == "__main__":
    # generate_train("training_data.csv", 10)
    # generate_input()
    #generate_train("validation_data.csv", 1000)
    plot_gestures()
