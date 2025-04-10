import streamlit as st
st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import warnings

# Suppress deprecation warnings globally
warnings.filterwarnings("ignore", category=FutureWarning)

st.title("CIFAR-10 Image Classifier")
st.markdown("Upload an image and see the predicted class, along with 5 similar CIFAR-10 samples that the model also predicts as the same class.")

# Load model
model_choice = st.selectbox("Choose a model", ["resnet32"])
if model_choice == "resnet32":
    from Model.resnet_model import model, device
    from Checkpoints import resnet32  # Only if needed elsewhere

    checkpoint_path = 'Checkpoints/resnet32.ckpt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼 Uploaded Image", width=200)

    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    pred_class = classes[predicted.item()]
    st.success(f"✅ Predicted Class: **{pred_class.upper()}**")

    # Load test set without transforms to keep images as PIL
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=None)

    # Find 5 test images that the model also predicts as the same class
    predicted_class_idx = classes.index(pred_class)
    matching_images = []

    indices = list(range(len(test_set)))
    random.shuffle(indices)

    for idx in indices:
        img, _ = test_set[idx]
        img_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_input)
            _, pred = torch.max(output, 1)
        if pred.item() == predicted_class_idx:
            matching_images.append(img)
        if len(matching_images) == 5:
            break

    st.markdown(f"### 🔍 5 CIFAR-10 Test Images Predicted as: `{pred_class}`")

    cols = st.columns(5)
    for i in range(5):
        cols[i].image(matching_images[i], caption=pred_class, use_container_width=True)
