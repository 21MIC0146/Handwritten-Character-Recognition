import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load the dataset
emnist_data = pd.read_csv('C:/A_Z Handwritten Data.csv').astype('float32')

# Split features and labels
x = emnist_data.drop('0', axis=1)
y = emnist_data['0']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Reshape and normalize data
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28, 1)) / 255.0
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28, 1)) / 255.0

# Convert labels to one-hot encoded vectors
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='valid'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Plot some sample predicted characters
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    predicted_label = np.argmax(model.predict(x_test[i:i+1]))
    true_label = np.argmax(y_test[i])
    color = 'green' if predicted_label == true_label else 'red'
    plt.xlabel(f"Predicted: {chr(predicted_label + 65)}, True: {chr(true_label + 65)}", color=color)
plt.tight_layout()
plt.show()

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title("Training vs Validation Accuracy")
plt.show()

# Save the trained model
model.save('alphabet_model.h5')
print("Model saved as 'alphabet_model.h5'")
