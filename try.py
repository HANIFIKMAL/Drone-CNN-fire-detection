import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import collections
import os
import streamlit as st
from PIL import Image
from torch.autograd import Variable
from pushbullet import Pushbullet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Streamlit UI
st.title("Fire, Smoke, and Neutral Image Detection App")

def load_image(image_up):
    img = Image.open(image_up)
    return img

# Main Streamlit app
if __name__ == "__main__":
    st.title("Image Uploader App")

image_up = st.file_uploader("Upload A Picture Here", type=['png', 'jpg', 'jpeg'])

if image_up is not None:
    st.image(image_up)
    file_details = {"FileName": image_up.name, "FileType": image_up.type}

    # Save the uploaded image
    with open(os.path.join("tempDir", image_up.name), "wb") as f:
        f.write(image_up.getbuffer())

    # Model loading and preprocessing
    model = torch.load("fire-flame.pt", map_location='cpu')  # Ensure the model is loaded to CPU

    model.eval()  # Set the model to evaluation mode

    dummy_input = torch.randn(1, 3, 224, 224)  # Dummy input for ONNX export

    # Export the model to ONNX format
    try:
        torch.onnx.export(
            model,               # The model to export
            dummy_input,         # Example input tensor
            "fire-flame.onnx",   # Output file name
            export_params=True,  # Export the model parameters
            opset_version=11,    # ONNX opset version
            do_constant_folding=True,  # Enable constant folding for optimization
            input_names=['input'],  # Input node names
            output_names=['output'],  # Output node names
            verbose=True         # Enable verbose logging
        )
        st.success("Model exported to ONNX format successfully!")
    except Exception as e:
        st.error(f"Failed to export model to ONNX: {e}")
        st.text(f"Error details: {e}")

# Image preprocessing
transformer = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if image_up is not None:
    img = Image.open(image_up)
    img_processed = transformer(img).unsqueeze(0)
    img_var = Variable(img_processed, requires_grad=False)
    img_var = img_var.to(device)  # Move the image to the correct device (GPU/CPU)

    # Predict the class using the trained model
    with torch.no_grad():
        logp = model(img_var)
        expp = torch.softmax(logp, dim=1)
        confidence, clas = expp.topk(1, dim=1)
        co = confidence.item() * 100
        class_no = str(clas.item())  # Extract class number

    # Image post-processing and displaying results
    orig = np.array(img)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (800, 500))

    # Display the prediction on the image
    if class_no == '1':
        label = f"Neutral: {co:.2f}%"
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        st.write(label)
    elif class_no == '2':
        label = f"Smoke: {co:.2f}%"
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        st.write(label)
    elif class_no == '0':
        label = f"Fire: {co:.2f}%"
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        st.write(label)

    # Save output image
    output_path = os.path.join("tempDir", "output_" + image_up.name)
    cv2.imwrite(output_path, orig)

    # Display the processed image in Streamlit
    st.image(output_path, caption="Processed Image", use_container_width=True)

# Alert buttons to notify authorities
st.header("ALERT AUTHORITY HERE")
fireButton = st.button("Fire Button")
smokeButton = st.button("Smoke Button")

API_KEY = "o.sKpoGn1um8K7f4r0Xn02B5VreRcquuKo"
file1 = "fire.txt"
file2 = "smoke.txt"

if fireButton:
    with open(file1, mode='r') as fire_file:
        fire_message = fire_file.read()
        pb = Pushbullet(API_KEY)
        pb.push_note('ALERT', fire_message)

elif smokeButton:
    with open(file2, mode='r') as smoke_file:
        smoke_message = smoke_file.read()
        pb = Pushbullet(API_KEY)
        pb.push_note('ALERT', smoke_message)
