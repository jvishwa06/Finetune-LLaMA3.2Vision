import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import requests
from io import BytesIO

model_name = "JVishu/Llama-3.2-11B-Vision-Radiology-mini"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_description(image, instruction):
    if isinstance(image, str):
        response = requests.get(image)
        img = Image.open(BytesIO(response.content))
    else:
        img = image

    inputs = tokenizer(
        instruction,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    inputs['pixel_values'] = img  
    outputs = model.generate(**inputs, max_length=128, temperature=1.5, use_cache=True)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

interface = gr.Interface(
    fn=generate_description,
    inputs=[
        gr.inputs.Image(type="pil", label="Upload Image"),
        gr.inputs.Textbox(label="Instruction", default="You are an expert radiographer. Describe accurately what you see in this image.")
    ],
    outputs="text",
    live=True,
    title="Radiology Image Description",
    description="Upload a radiology image and provide an instruction to generate a description based on the image."
)

interface.launch()
