# train_mnist_augmented.py
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------
# 1️⃣ Load MNIST data
# ---------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ---------------------------
# 2️⃣ Data augmentation
# ---------------------------
datagen = ImageDataGenerator(
    rotation_range=15,        # rotate +-15 degrees
    width_shift_range=0.1,    # shift horizontally
    height_shift_range=0.1,   # shift vertically
    zoom_range=0.1             # zoom in/out
)

datagen.fit(x_train)

# ---------------------------
# 3️⃣ Build CNN model
# ---------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------
# 4️⃣ Train model with augmented data
# ---------------------------
batch_size = 128
epochs = 10

model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          steps_per_epoch=len(x_train)//batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# ---------------------------
# 5️⃣ Evaluate model
# ---------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# ---------------------------
# 6️⃣ Save model
# ---------------------------
model.save("mnist_cnn_augmented.h5")
print("Model saved as mnist_cnn_augmented.h5")
