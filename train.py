from data_loader import load_data
from model import create_model

(x_train, y_train), (x_test, y_test) = load_data()
model = create_model()

model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

model.save("cnn_cifar10_model.h5")
plt.savefig("accuracy_graph.png")
plt.show()