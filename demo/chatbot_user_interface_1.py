import gradio as gr
import random

def random_response(message, history):
    return random.choice(["Yes", "No"])

gr.ChatInterface(
    fn=random_response, 
    type="messages"
).launch()