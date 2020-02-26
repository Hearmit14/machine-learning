import tensorflow as tf
import pandas as pd
from tensorflow import keras

train = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice//mnist/train.csv')
train.shape

test = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice//mnist/test.csv')
test.shape

# y_train = train['label']
# X_train = train.drop('label', 1)
# X_test = test

# X_train.info()
# x_train.index()
# X_train[:1]
# X_test.info()


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train[:1]
x_train.shape

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))

model.summary()


model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model1.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
