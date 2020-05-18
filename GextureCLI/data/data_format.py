import os 
import numpy as np
import math
import random
from pathlib import Path

data = np.zeros(0)
PATH = Path(os.path.dirname(os.path.realpath(__file__)))

def append_to_data(sample):
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
            function(t) + random.random()/10
        ] for t in range(200)])
    return data_tmp

for i in range(10):
    data = append_to_data(generate_sample(lambda x: math.cos(x), 1, i))
    data = append_to_data(generate_sample(lambda x: math.sqrt(x), 2, i+10))


np.savetxt(str(PATH / "training_data.csv"), data, fmt='%s', delimiter=",")
