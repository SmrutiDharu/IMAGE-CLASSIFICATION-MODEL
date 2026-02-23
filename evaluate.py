import tensorflow as tf
from data_loader import load_data

(_, _), (x_test, y_test) = load_data()

model = tf.keras.models.load_model("cnn_cifar10_model.h5")
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)