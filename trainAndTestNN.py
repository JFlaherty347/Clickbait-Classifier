import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers 
from keras.models import Sequential

from keras.utils import to_categorical
import kerastuner as kt

activation_functions = ['relu', 'elu']
output_activation_functions = ['sigmoid', 'softmax']
optimizers = {'adam': tf.keras.optimizers.Adam, 'RMSProp': tf.keras.optimizers.RMSprop, 'SGD': tf.keras.optimizers.SGD}
learning_rates = [1e-2, 1e-3, 1e-4]

def print_scores(model, scores):
	print("\n")
	for i in range(len(scores)):
		print(model.metrics_names[i], " = ", scores[i])

def model_builder(hp):
  first_layer = hp.Int('first_layer_hp', min_value=50, max_value=150, step=25)
  second_layer = hp.Int('second_layer_hp', min_value=50, max_value=150, step=25)
  third_layer = hp.Int('third_layer_hp', min_value=50, max_value=150, step=25)

  hidden_layer_activation = hp.Choice('hidden_layer_activation', values=activation_functions)
  output_layer_activation = hp.Choice('output_layer_activation', values=output_activation_functions)

  optimizer = hp.Choice('optimizer', values=['adam', 'RMSProp', 'SGD'])

  learning_rate = hp.Choice('lr', values=learning_rates)

  model = keras.Sequential()
  #model.add(layers.Flatten())
  model.add(layers.Dense(first_layer, activation=hidden_layer_activation))
  model.add(layers.Dense(second_layer, activation=hidden_layer_activation))
  model.add(layers.Dense(third_layer, activation=hidden_layer_activation))
  model.add(layers.Dense(2, activation=output_layer_activation))

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
  model.compile(optimizer=optimizers[optimizer](learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy', 'categorical_crossentropy', ])

  return model


def tune_nn_model(x_train, y_train):
  y_train = to_categorical(y_train)
  tuner = kt.BayesianOptimization(model_builder, objective='val_accuracy', max_trials=4)
  stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

  tuner.search(x_train, y_train, epochs=50, validation_split=0.3, callbacks=[stop_early])

  best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

  print("Best Hyperparams:", best_hps)

  # Calculate best epochs
  model = tuner.hypermodel.build(best_hps)
  history = model.fit(x_train, y_train, epochs=50, validation_split=0.3)
  val_acc_per_epoch = history.history['val_accuracy']
  best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

  # Train for said epochs
  model = tuner.hypermodel.build(best_hps)
  model.fit(x_train, y_train, epochs=best_epoch, validation_split=0.3)

  #Save model
  model.save("simpleNNBestModel.h5")


def test_simple_fc_nn(x_test, y_test):
  y_test = to_categorical(y_test)
  best_simple_NN_model = keras.models.load_model('simpleNNBestModel.h5')
  # best_simple_NN_model = create_simple_NN(activation_fn = best_simple_NN_config[1], output_activation_fn = best_simple_NN_config[2],
  # optimizer = best_simple_NN_config[3], learning_rate = best_simple_NN_config[4]) 

  # best_simple_NN_model.fit(x_train, y_train, epochs=20, verbose=1)
  scores = best_simple_NN_model.evaluate(x_test, y_test, verbose=1)
  print("\n\nBest Model Scores on Test Data: ")
  print_scores(best_simple_NN_model, scores)

  prediction = best_simple_NN_model.predict(x_test)
  n_values = 2; 
  prediction = np.eye(n_values, dtype=int)[np.argmax(prediction, axis=1)]
  class_report = classification_report(y_test, prediction)
  print("Report: \n", class_report)
  conf_matrix = confusion_matrix(y_test.argmax(axis=1), prediction.argmax(axis=1))
  print("\nConfusion Matrix:\n", conf_matrix)
  f1 = f1_score(y_test.argmax(axis=1), prediction.argmax(axis=1))
  print("\nF1-Score: ", f1)

