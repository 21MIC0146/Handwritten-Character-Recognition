import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Define the path to the dataset
dataset_path = "C:/curated"

# Load the dataset
data = []
labels = []

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        label = folder_name
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Resize the image to 28x28 pixels
            image = cv2.resize(image, (28, 28))
            # Normalize pixel values to the range [0, 1]
            image = image / 255.0
            data.append(image)
            labels.append(label)

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Convert labels to numerical format
unique_labels = np.unique(labels)
label_to_index = {label: i for i, label in enumerate(unique_labels)}
index_to_label = {i: label for label, i in label_to_index.items()}
numerical_labels = np.array([label_to_index[label] for label in labels])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, numerical_labels, test_size=0.2, random_state=42)

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(unique_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Make predictions
predictions = model.predict(X_test)

# Plot some predicted and actual characters
plt.figure(figsize=(15, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    color = 'green' if predicted_label == true_label else 'red'
    plt.xlabel(f"Predicted: {index_to_label[predicted_label]}\nTrue: {index_to_label[true_label]}", color=color)

plt.tight_layout()
plt.show()
