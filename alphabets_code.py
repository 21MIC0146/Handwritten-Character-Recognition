import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('alphabet_model.h5')

# Load and process the image
image_path = "C:/Users/sravani/OneDrive/Desktop/SEM 6/SOFT COMPUTING/PROJECT/SAMPLE_LETTERS.png"
image = Image.open(image_path).convert("L")  # Convert to grayscale
image = image.resize((28 * 4, 28))  # Assuming 4 characters horizontally
image_array = np.array(image)

# Split the image into individual characters
characters = []
for i in range(4):
    char_image = image_array[:, i*28:(i+1)*28]
    characters.append(char_image)

# Prepare characters for prediction
predicted_word = ""
for char_image in characters:
    char_image = char_image.reshape(1, 28, 28, 1) / 255.0  # Normalize
    predicted_label = model.predict(char_image)
    predicted_alphabet = chr(np.argmax(predicted_label) + 65)  # Map to 'A'-'Z'
    predicted_word += predicted_alphabet.upper()

# Save the predicted word as text
output_path = "C:/Users/sravani/OneDrive/Desktop/SEM 6/SOFT COMPUTING/PROJECT/predicted_word.txt"
with open(output_path, "w") as file:
    file.write(predicted_word)

# Output results
print("Predicted word:", predicted_word)
print("Result saved at:", output_path)
