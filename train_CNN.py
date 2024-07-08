import tensorflow as tf
from keras import models, layers


def create_and_save_digit_recognition_model():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Load and preprocess MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )
    train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

    # Train the model
    model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

    # Save the model
    model.save("digit_recognition_model.h5")


# Run this function once to create and save the model
create_and_save_digit_recognition_model()
