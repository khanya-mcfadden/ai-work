import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import kerastuner as kt
 
print("TensorFlow version:", tf.__version__)
print("Keras Tuner version:", kt.__version__)
 
# Load and preprocess the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
 
# Data augmentation
data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomRotation(0.2),
    keras.layers.experimental.preprocessing.RandomZoom(0.2),
])
 
# Display the number of GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
 
# Define the model builder function for hyperparameter tuning
def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    # Adding data augmentation layers
    model.add(keras.layers.Reshape((28, 28, 1)))
    model.add(data_augmentation)
    model.add(keras.layers.Flatten())
    hp_units = hp.Int('units', min_value=1, max_value=512, step=10)
    model.add(keras.layers.Dense(units=hp_units, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10))
 
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
 
    return model
 
# Set up the hyperparameter tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')
 
# Early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
 
# Perform the hyperparameter search
tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
 
# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
 
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
 
# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
 
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
 
# Retrain the model with the optimal number of epochs
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)
 
# Evaluate the model on the test data
eval_result = hypermodel.evaluate(x_test, y_test)
print("[test loss, test accuracy]:", eval_result)
 
# Make predictions
predictions = hypermodel.predict(x_test)
 
# Display a prediction for the first test image
plt.figure()
plt.imshow(x_test[0])
plt.grid(False)
plt.show()
 
# Save and reload the model
hypermodel.save('my_model.keras')
new_model = tf.keras.models.load_model('my_model.keras')
new_predictions = new_model.predict(x_test[:6])
print(np.argmax(new_predictions, axis=1))