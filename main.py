
import gradio as gr
from fastai.vision.all import *

# Load the trained model (make sure this path is correct or load from external URL if hosted)
learn = load_learner('food_classifier.pkl')

# Define the prediction function
def predict_food(image):
    image = image.resize((224, 224))  # Resize the image to match the input size of the model
    pred, pred_idx, probs = learn.predict(image)
    return {str(learn.dls.vocab[i]): float(probs[i]) for i in range(len(probs))}

# Set up the Gradio interface
image_input = gr.Image(type="pil")
label_output = gr.Label(num_top_classes=3)

interface = gr.Interface(fn=predict_food, 
                         inputs=image_input, 
                         outputs=label_output, 
                         title="Food Classifier", 
                         description="Upload a food image to classify it into one of 21 categories.")

# Launch the app
interface.launch(share=True)  # 'share=True' generates a public link to share the app
