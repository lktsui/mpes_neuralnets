#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import everything needed
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


# In[2]:


# Import the data
data = pd.read_csv('NH3_NOx_data.txt', sep=",")
data.columns = ['NO','NO2','CO','NH3','V0','V1','V2','V3','V4','V5','tf']

# Remove outliers
    # In the script "Analysis of Old Data (Plots and Data Features)" I identified several outliers which I will now drop
data = data.drop(80)

#create a NOx column
NOx = pd.DataFrame(data['NO']+data['NO2'], columns=['NOx'])
columns = [NOx, data]
data = pd.concat(columns, axis=1)

#drop the CO, tf, NO, NO2, and NH3 columns
data = data.drop(['CO','tf','NO','NO2','NH3','V0','V3'],axis=1)

#scramble the dataset and reindex (this prevents the model from learning relationships over time)
data = data.reindex(np.random.permutation(data.index))
data = data.reset_index(drop=True)

#standardize the data by calculating the z-scores of just the voltages
data_mean = data.mean()
data_std = data.std()
data_norm = (data - data_mean)/data_std
data_norm = data_norm.drop(['NOx'], axis=1)

#standardize the NOx concentrations by dividing all of them by 300 
data_norm_c = data.drop(['V1','V2','V4','V5'],axis=1)
data_norm_c = data_norm_c/300

#add the concentration dataframe to the z-score dataframe
columns = [data_norm_c, data_norm]
data = pd.concat(columns, axis=1)

#fill any NaN values with a value of zero
data = data.fillna(0)


# In[3]:


## This cell defines and calls the function divide_data
    
# divide_data allows for easy changing in the divide between training and testing data
def divide_data(percent,df):
    #find the number of rows in the dataframe
    row, column = df.shape
    
    #multiply that by the given percent and round up
    end = round(percent*row)
    
    #make the training data a new dataframe from 0 to the variable end (uninclusive)
    df_train = df[0:end]
    
    #make the test data a new dataframe from the variable end to the end of the original dataframe and reindex
    df_test = df[end:]
    df_test = df_test.reset_index(drop=True)
    
    #return the new dataframes so they can be accessed upon calling the function
    return df_train, df_test

#call on divide_data to segment the data into train and test according to the variable percent
    #the value of percent can be changed for optomization purposes
percent = .6
data_train, data_test = divide_data(percent,data)


# In[4]:


## make feature list of everything but the gas concentrations

#create empty variable feature_columns
feature_columns = []

#make a variable for each of the 6 sensors and append to feature_columns
#V0_voltage = tf.feature_column.numeric_column("V0")
#feature_columns.append(V0_voltage)

V1_voltage = tf.feature_column.numeric_column("V1")
feature_columns.append(V1_voltage)

V2_voltage = tf.feature_column.numeric_column("V2")
feature_columns.append(V2_voltage)

#V3_voltage = tf.feature_column.numeric_column("V3")
#feature_columns.append(V3_voltage)

V4_voltage = tf.feature_column.numeric_column("V4")
feature_columns.append(V4_voltage)

V5_voltage = tf.feature_column.numeric_column("V5")
feature_columns.append(V5_voltage)

#combine into one feature layer using feature_columns
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# In[5]:


## create a function that creates the deep neural net model
    # this code is primarily taken from the Google TF tutorial, but I actually understand what it does this time
def create_model(my_learning_rate, my_feature_layer):
    #zero out the model
    model = None
    
    #set early stopping to enact if the loss to improve for 5 epochs
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=20)
    
    #the model will be sequential
    model = tf.keras.models.Sequential()

    #stop an error from popping up
    tf.keras.backend.set_floatx('float64')

    #add the layer containing the feature columns to the model
    model.add(my_feature_layer)

    #create hidden layer(s)
    model.add(tf.keras.layers.Dense(units=36, 
                                  activation='sigmoid', #kernel_regularizer=tf.keras.regularizers.l1(1.0),
                                  name='Hidden1'))
    
    model.add(tf.keras.layers.Dense(units=36, 
                                  activation='sigmoid', #kernel_regularizer=tf.keras.regularizers.l1(1.0), 
                                  name='Hidden2'))
    
    #model.add(tf.keras.layers.Dropout(rate=1.0))
    
    #define the output layer with one output
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid',  
                                  name='Output'))        
    
    #compile the model using the TF Adam optimizer
        #I experimented with other optimizers that TF has and Adam seemed to give the best results out of all of them
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])

    #when the function is called it returns the created model
    return model, early_stop


# In[6]:


#define the function train_model
    #again, this is a function adopted from the google TF tutorial
def train_model(model, dataset, epochs, label_name, early_stop, batch_size=None):
    #split the dataset into features and labels
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name[0]))
    history = model.fit(x=features, y=[label], batch_size=batch_size,
                      epochs=epochs, callbacks=[early_stop], shuffle=True) 

    #the  list of epochs is stored separately from the rest of history.
    epochs = history.epoch
  
    #to track the progression of training, gather a snapshot
    #of the model's mean squared error at each epoch. 
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]

    return epochs, mse, history.history


# In[7]:


# The following variables are the hyperparameters.
learning_rate = 0.05
epochs_num = 500
batch_size = 62
#validation_split = 0.2

# Specify the label
label_name = ["NOx"]

# Establish the model's topography.
my_model = None
my_model, early_stop = create_model(learning_rate, my_feature_layer)

# Train the model on the normalized training set. We're passing the entire
# normalized training set, but the model will only use the features
# defined by the feature_layer.
epochs, mse, history = train_model(my_model, data_train, epochs_num, 
                          label_name, early_stop, #validation_split, 
                                   batch_size)

#print whether the model stopped training early and what epoch it stopped at if early stopping was used
if len(history["mean_squared_error"]) == epochs_num:
    print('The model did not stop training early.')
    
if len(history["mean_squared_error"]) != epochs_num:
    print('The model stopped training at',len(history['loss']),'epochs because of early stopping.')

# After building a model against the training set, test that model
# against the test set.
test_features = {name:np.array(value) for name, value in data_test.items()}
train_features = {name:np.array(value) for name, value in data_train.items()}
test_label = np.array(test_features.pop(label_name[0]))

print("\n Evaluate the new model against the test set:")

my_test = my_model.evaluate(x = test_features, y = [test_label], batch_size=batch_size) 

