import matplotlib.pyplot as plt
from model import create_model
from data_loader import load_data

(x_train, y_train), (x_test, y_test) = load_data()
model = create_model()

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy vs Epochs")
plt.show()