import streamlit as st
from PIL import Image
import numpy as np
from imageCompression import load_and_resize_image, compress_image
import matplotlib.pyplot as plt
from io import BytesIO
import requests

def main():
    st.title("Image Compression App")

    upload_option = st.radio("Select the upload method:", ("Upload from Computer", "Upload from URL"))
    
    image = handle_upload(upload_option)

    if image is not None:
        np_img = np.array(image)
        resized_img = load_and_resize_image(np_img)

        if st.columns(3)[1].button("Compress Image", use_container_width=True):
            compressed_img = compress_image(resized_img, 16)
            display_compressed_image(compressed_img)
            provide_download_link(compressed_img)
            # st.info("Upload another image to compress.")

def handle_upload(upload_option):
    if upload_option == "Upload from Computer":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            return Image.open(uploaded_file)
    elif upload_option == "Upload from URL":
        url_input = st.text_input("Enter an image URL")
        if url_input:
            return load_image_from_url(url_input)
    return None

def load_image_from_url(url):
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error("Error loading image from URL. Please check the URL and try again.")
        return None

def display_compressed_image(compressed_img):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(compressed_img)
    ax.set_title("Compressed Image")
    ax.axis("off")
    st.pyplot(fig)

def provide_download_link(compressed_img):
    compressed_img_bytes = BytesIO()
    plt.imsave(compressed_img_bytes, compressed_img, format='png')
    compressed_img_bytes.seek(0)
    st.download_button(label="Download Compressed Image", data=compressed_img_bytes, file_name='compressed_image.png', mime='image/png')

if __name__ == "__main__":
    main()
