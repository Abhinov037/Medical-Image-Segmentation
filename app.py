import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('vit_brain_tumor_segmentation.h5',
                                       custom_objects={'dice_coef': dice_coef, 'combined_loss': combined_loss})
    return model

# Custom loss and metric used during training
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.6 * dice + 0.4 * bce

# Image preprocessing
def preprocess_image(uploaded_image):
    image = uploaded_image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# App UI
st.title("🧠 Brain Tumor Segmentation using ViT")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner('Segmenting tumor...'):
        model = load_model()
        preprocessed = preprocess_image(image)
        prediction = model.predict(preprocessed)[0, :, :, 0]

        # Binarize the prediction for visualization
        binary_mask = (prediction > 0.5).astype(np.uint8) * 255

        # Show result
        st.subheader("Prediction (Tumor Mask):")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(binary_mask, cmap='gray')
        ax[1].set_title("Predicted Mask")
        ax[1].axis("off")

        st.pyplot(fig)
