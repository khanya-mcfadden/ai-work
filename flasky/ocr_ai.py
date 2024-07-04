# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)

# %%
import kerastuner as kt
print("Keras Tuner version:", kt.__version__)

# %%
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# %%
mnist
(x_train1, y_train1), (x_test1, y_test1) = mnist.load_data()
print(x_train)

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# %%
def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  hp_units = hp.Int('units', min_value = 1, max_value = 512, step = 10)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

# %%
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

# %%
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# %%
tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# %%


# %%
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
#model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# %%
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)

# %%
x_test.shape

# %%
predictions = model.predict(x_test)
predictions

# %%
model.summary()

# %%
x_test[:1].shape

# %%
pred=tf.nn.softmax(predictions).numpy()
pred[0]

# %%
y_pred = np.argmax(pred[0])
y_pred

# %%
plt.figure()
plt.imshow(x_test[0])
plt.grid(False)
plt.show()

# %%
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# %%
score = model.evaluate(x_test,  y_test, verbose=2)

# %%
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# %%
probability_model(x_test[:5])

# %%
print(score)


# %%
eval_result = hypermodel.evaluate(x_test, y_test)
print("[test loss, test accuracy]:", eval_result)

# %%
model.save('my_model.keras')

# %%
newmodel = tf.keras.models.load_model('my_model.keras')
newpred = newmodel.predict(x_test[:6])
print(np.argmax(newpred))


