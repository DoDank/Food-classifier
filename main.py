import gradio as gr
from fastai.vision.all import *

# Load the trained model
learn = load_learner('food_classifier.pkl')

# Define a function that makes predictions using the model
def predict_food(image):
    image = image.resize((224, 224))  # Resize the image to match the input size of the model
    pred, pred_idx, probs = learn.predict(image)
    return {str(learn.dls.vocab[i]): float(probs[i]) for i in range(len(probs))}

# Define Gradio interface using the updated API
image_input = gr.Image(type="pil")  # Removed the 'shape' argument
label_output = gr.Label(num_top_classes=3)

# Create the Gradio app
interface = gr.Interface(fn=predict_food,
                         inputs=image_input,
                         outputs=label_output,
                         title="Food Classifier",
                         description="Upload a food image to classify it into one of 21 categories.")

# Launch the app
interface.launch()
