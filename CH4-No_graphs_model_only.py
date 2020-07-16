# Import needed items

try:
    import tensorflow as tf
    from tensorflow import keras
    tflite = tf.lite

except:
    import tflite_runtime.interpreter as tflite

import pandas as pd

import numpy as np
import time
import os

data_1 = pd.read_csv(os.path.join('data', 'CH4_N2O.txt'), sep=",")
data_1.columns = ['CH4', 'N2O', 'NO', 'NH3', 'CH4_NG_CAL', 'V0', 'V1', 'time']

data_2 = pd.read_csv(os.path.join('data', 'CH4_NH3.txt'), sep=",")
data_2.columns = ['CH4', 'N2O', 'NO', 'NH3', 'CH4_NG_CAL', 'V0', 'V1', 'time']

data_3 = pd.read_csv(os.path.join('data', 'CH4 CH4_NG_CAL.txt'), sep=",")
data_3.columns = ['CH4', 'N2O', 'NO', 'NH3', 'CH4_NG_CAL', 'V0', 'V1', 'time']

combined = [data_1, data_2, data_3]
data = pd.concat(combined, axis=0)
data = data.reset_index(drop=True)

# standardize the voltage data
data_v = data.drop(['CH4', 'N2O', 'NO', 'NH3', 'CH4_NG_CAL', 'time'], axis=1)
data_mean = data_v.mean()
data_std = data_v.std()
data_norm = (data_v - data_mean) / data_std

# Create a new classification column with appropriate values for each concentration combination

class_column = data.drop(['CH4', 'N2O', 'NH3', 'CH4_NG_CAL', 'V0', 'V1', 'time'], axis=1)
class_column.columns = ['Class']

class_column[18:19] = 0
class_column[37:38] = 0
class_column[56:57] = 0
class_column[75:76] = 0
class_column[94:95] = 0
class_column[113:114] = 0
class_column[132:133] = 0
class_column[151:152] = 0
class_column[188:189] = 0
class_column[207:208] = 0
class_column[226:227] = 0
class_column[245:246] = 0
class_column[264:265] = 0
class_column[283:284] = 0
class_column[302:303] = 0
class_column[321:337] = 0

# N2O
class_column[0:18] = 1

# NH3
class_column[170:188] = 2

# CH4+N2O
class_column[19:37] = 3
class_column[38:56] = 3
class_column[57:75] = 3
class_column[76:94] = 3
class_column[95:113] = 3
class_column[114:132] = 3
class_column[133:151] = 3
class_column[152:170] = 3

# CH4+NH3
class_column[189:207] = 4
class_column[208:226] = 4
class_column[227:245] = 4
class_column[246:264] = 4
class_column[265:283] = 4
class_column[284:302] = 4
class_column[303:321] = 4

# CH4_NG_CAL
class_column[337:] = 5

# combine the class column and standardized data into one dataframe
columns = [class_column, data_norm]
data = pd.concat(columns, axis=1)

data = data.drop(index=[56, 75, 94, 113, 132, 151, 188, 207, 226, 245, 264, 283, 302])
data = data.drop(index=[76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93])

# scramble the data
data = data.reindex(np.random.permutation(data.index))
data = data.reset_index(drop=True)

# split the data back into separate classification and sensor data
class_column = data.drop(['V0', 'V1'], axis=1)
data_norm = data.drop(['Class'], axis=1)


## This cell defines and calls the function divide_data

# divide_data allows for easy changing in the divide between training and testing data
def divide_data(percent, df):
    # find the number of rows in the dataframe
    row, column = df.shape

    # multiply that by the given percent and round up
    end = round(percent * row)

    # make the training data a new dataframe from 0 to the variable end (uninclusive)
    df_train = df[0:end]

    # make the test data a new dataframe from the variable end to the end of the original dataframe and reindex
    df_test = df[end:]
    df_test = df_test.reset_index(drop=True)

    # return the new dataframes so they can be accessed upon calling the function
    return df_train, df_test


#set parameters
epochs = 500
patience = 5
percent = .80

test_batch = 8
threshold = 0

#call on divide_data to segment both the sensor and class data into train and test according to the variable percent
sensor_train, sensor_test = divide_data(percent, data_norm)
class_train, class_test = divide_data(percent, class_column)

# model = keras.models.load_model(os.path.join('models', 'ch4_model'))

test_dataset = []
train_dataset = []
true_result = class_test.values

for test_data_point in sensor_test.values:
    test_dataset.append(np.reshape(test_data_point, (-1,2)).astype(np.float32))

for train_data_point in sensor_train.values:
    train_dataset.append(np.reshape(train_data_point, (-1,2)).astype(np.float32))

interpreter = tflite.Interpreter(model_path=os.path.join('models', 'ch4_model.tflite'))
interpreter.allocate_tensors()

output_details = interpreter.get_output_details()

start = time.time()
for i in range(0,1000):
    for ind, test_point in enumerate(train_dataset):
        interpreter.set_tensor(1, test_point)
        interpreter.invoke()
        output_data = np.argmax(interpreter.get_tensor(output_details[0]['index']))

    for ind, test_point in enumerate(test_dataset):
        interpreter.set_tensor(1, test_point)
        interpreter.invoke()
        output_data = np.argmax(interpreter.get_tensor(output_details[0]['index']))

# stop timing
end = time.time()

data_size = 1000*(len(test_dataset)+len(train_dataset))

delta_time = end-start
time_per_pt = delta_time/data_size

print('\n Testing took ' + "%.2f " % (delta_time), 'seconds. (%0.2e s per point)'%time_per_pt)






