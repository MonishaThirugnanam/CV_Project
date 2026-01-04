# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf

CLASSES = ["glass", "paper", "plastic", "metal"]
IMG_SIZE = (224,224)

st.title("Waste Classification Web App")

# Load pre-trained model
model = load_model("waste_model.h5")

# -------------------------------
# Grad-CAM FUNCTION
# -------------------------------
def get_gradcam(model, img_array, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model([model.inputs], 
                                       [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array_exp = np.expand_dims(img_array, axis=0)
    heatmap = get_gradcam(model, img_array_exp)
    
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(cv2.cvtColor(np.uint8(img_array*255), cv2.COLOR_RGB2BGR), 
                                      0.6, heatmap_color, 0.4, 0)
    st.image(superimposed_img, channels="BGR", caption="Grad-CAM Heatmap")

# -------------------------------
# STREAMLIT IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array_exp = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array_exp)
    class_idx = np.argmax(pred)
    st.write(f"Predicted Class: {CLASSES[class_idx]} (Confidence: {pred[0][class_idx]*100:.2f}%)")
    
    display_gradcam(uploaded_file)
