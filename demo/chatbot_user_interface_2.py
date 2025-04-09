import gradio as gr
import random
import importlib
import lawbuddy.rag  # Import the module first
import openai
import os
from dotenv import load_dotenv


importlib.reload(lawbuddy.rag)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

pipeline = lawbuddy.rag.SimpleRagPipeline.from_api(
    model="typhoon-v2-70b-instruct",
    api_base="https://api.opentyphoon.ai/v1",
    context_window=8192,
    is_chat_model=True,
    max_tokens=768*2,
    is_function_calling_model=False,
    api_key=os.getenv("TYPHOON_API_KEY")
)

pipeline.load_vector_store(path="spaces/simple_rag_student_rules")

def chat(message, history):
    for token in pipeline.query(message, verbose=True, streaming = True):
        yield token[1]

gr.ChatInterface(
    fn=chat, 
    type="messages"
).launch()