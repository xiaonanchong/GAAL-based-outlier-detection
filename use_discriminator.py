import tensorflow as tf
from scipy.io import loadmat
import numpy as np
new_model = tf.keras.models.load_model('trained_discriminator.h5')

def load_data():
    slice_len = 500
    data = loadmat('/home/anjie/data/97.mat')['X097_DE_time']
    l = int(data.shape[0]/slice_len)*slice_len
    data_x = np.array(data[:l]).reshape(-1, slice_len) 
    data_y = np.array(['nor' for i in range(data_x.shape[0])])
    return data_x, data_y

x, y = load_data()
r = new_model.predict(x)>0.99

key = np.unique(r)
result = {}
for k in key:
    mask = (r == k)
    r_new = r[mask]
    v = r_new.size
    result[k] = v
print(result)
print(1-float(result[False])/(result[False]+result[True]))


