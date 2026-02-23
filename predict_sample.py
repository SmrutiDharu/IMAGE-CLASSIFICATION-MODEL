import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_data

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

(_, _), (x_test, y_test) = load_data()
model = tf.keras.models.load_model("cnn_cifar10_model.h5")

predictions = model.predict(x_test[:5])

plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[i], interpolation='nearest')
    plt.title(class_names[predictions[i].argmax()])
    plt.axis('off')

plt.show()