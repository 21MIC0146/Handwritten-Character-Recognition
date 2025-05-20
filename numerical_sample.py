import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np

# Load and preprocess MNIST data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Load and preprocess input image
img_path = "C:/Users/sravani/OneDrive/Desktop/SEM 6/SOFT COMPUTING/PROJECT/SAMPLE.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
_, threshold_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

predicted_digits = []

# Predict digits in the image
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    digit = threshold_img[y:y+h, x:x+w]
    resized_digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
    resized_digit = cv2.bitwise_not(resized_digit)
    normalized_digit = resized_digit / 255.0
    input_digit = np.expand_dims(normalized_digit, axis=0)
    prediction = model.predict(input_digit, verbose=0)
    predicted_digit = np.argmax(prediction)
    predicted_digits.append(str(predicted_digit))

# Save predicted digits to a file
predicted_digits_str = "".join(predicted_digits)
output_file_path = "C:/Users/sravani/OneDrive/Desktop/SEM 6/SOFT COMPUTING/PROJECT/predicted_digits.txt"
with open(output_file_path, "w") as file:
    file.write(predicted_digits_str)

print("Predicted digits saved in:", output_file_path)
