import os 
import numpy as np
import math
import random
from pathlib import Path
import matplotlib.pyplot as plt

"""
def generate_sample_from_func(function, sample_class, sample_id):
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
        data = custom_append(data, generate_sample_from_func(lambda x: math.cos(x/120) + random.random()/20, 1, i))
        data = custom_append(data, generate_sample_from_func(lambda x: math.sqrt(x)/math.sqrt(200) + random.random()/20, 2, i+10))
        data = custom_append(data, generate_sample_from_func(lambda x: math.exp(x/100)/math.exp(2) + random.random()/20, 3, i+20))

def generate_input():
    input_sample = generate_sample_from_func(lambda x: math.cos(x/120) + random.random()/20,1, 1)
    input_sample = input_sample[:, 2:]
    np.savetxt(str(PATH / "input_data.csv"), input_sample, fmt='%s', delimiter=",")


def plot_gestures():
    plt.figure(1)
    plt.plot([math.cos(x/120) for x in range(200)])
    plt.plot([math.sqrt(x)/math.sqrt(200) for x in range(200)])
    plt.plot([math.exp(x/100)/math.exp(2) for x in range(200)])
    plt.show()
"""

PATH = Path(os.path.dirname(os.path.realpath(__file__)))

def custom_append(data, sample):
    if data.size == 0:
        return sample
    if data.size == sample.size:
        return np.append([data], [sample], axis=0)
    elif len(sample.shape) == 2:
        return np.append(data, sample, axis=0)
    elif sample.size > 0:
        return np.append(data, [sample], axis=0)
    else:
        return data

def generate_sample_from_array(array, sample_class, sample_id):
    data_tmp = np.zeros(0)
    for element in array:
        data_tmp = custom_append(data_tmp, np.concatenate(([sample_id, sample_class], element)))
    return data_tmp

def format_real_files():
    gesture_type = ["uniform", "movement", "speed"]
    sample_id = 0
    temp_sample_array = np.zeros(0)
    last_label = 1
    data_array = np.zeros(0)
    for (idx, gtype) in enumerate(gesture_type):
        dirlist = os.listdir(str(PATH / "data_real" / gtype))
        for datafilename in dirlist:
            datafile = open(str(PATH / "data_real" / gtype / datafilename), 'r', encoding='utf-8')
            for line in datafile:
                if "label" not in line:
                    data_line_array = np.array([float(x) for x in line[:-1].split("\t")[1:]])
                    temp_sample_array = custom_append(temp_sample_array, data_line_array)
                elif len(temp_sample_array) > 0:
                    sample_id += 1
                    sample_array = generate_sample_from_array(temp_sample_array[4:], last_label, sample_id)
                    data_array = custom_append(data_array, sample_array)
                    temp_sample_array = np.zeros(0)
                    last_label = int(line[6]) + 1
        break
    np.savetxt(str(PATH / "prepared_real_data.csv"), data_array, fmt='%s', delimiter=",")

if __name__ == "__main__":
    # generate_train("training_data.csv", 10)
    #generate_input()
    #generate_train("validation_data.csv", 1000)
    #plot_gestures()
    format_real_files()
