from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model
model = load_model("jujube.h5")  # Ensure the model file is in the same directory or provide the correct path

# Define a dictionary to map class indices to class names
class_names = {
    0: "Bruised",  # Replace with your actual class names
    1: "Healthy",  # Replace with your actual class names
}

@app.route('/', methods=['GET'])
def home():
    # Render the HTML form for uploading images
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Handle the uploaded file
    imagefile = request.files['imagefile']
    image_path = os.path.join("static", imagefile.filename)  # Save to the static folder

    # Ensure the static directory exists
    if not os.path.exists("static"):
        os.makedirs("static")

    try:
        imagefile.save(image_path)
    except FileNotFoundError:
        return "Error: Unable to save the file. Check the static folder."

    # Preprocess the image to match the model's expected input size
    image = load_img(image_path, target_size=(180, 180))  # Ensure 180x180 size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction using the model
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest confidence
    confidence = predictions[0][predicted_class_index]  # Get the confidence score for the predicted class

    # Map the predicted class index to the class name
    predicted_class_name = class_names[predicted_class_index]

    # Construct the detailed classification result
    classification = f"Class {predicted_class_index}: {predicted_class_name} (Confidence: {confidence:.2f})"

    # Return the result to the user
    return render_template('index.html', prediction=classification, image_url=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
