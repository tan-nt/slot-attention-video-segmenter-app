import streamlit as st
import logging
from streamlit_option_menu import option_menu
from PIL import Image
import os
import os
import numpy as np
import cv2

import torch
from model.model_for_pretrain import GSANet


# Configure Streamlit page
st.set_page_config(
    page_title="Guided Slot Attention for Video Object Segmentation",
    page_icon=":movie_camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set up logging
logger = logging.getLogger()
logging.basicConfig(encoding="UTF-8", level=logging.INFO)

# Sidebar Navigation
st.sidebar.image("assets/slot-attention.jpg", use_container_width=True)
st.sidebar.header("Navigation")
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["📹 Introduction", "🛠️ Try It Out", "📊 Real-Life Applications", "📜 FAQs"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        },
    )

def visual(image, image_name, work_dir="log/2024-11-14 12:52:12-davis-2017"):
    # Set the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: {device}")

    # Load the model
    model = GSANet().to(device)
    model = torch.nn.DataParallel(model)

    # Load checkpoint
    model_dir = os.path.join(work_dir, "model")
    checkpoint_path = os.path.join(model_dir, "best_model.pth")
    
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint file not found: {checkpoint_path}")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return

    # Preprocess the uploaded image for the model
    try:
        image = image.convert("RGB")  # Ensure the image is in RGB format
        image = np.array(image)
        image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W) tensor
        image = image / 255.0  # Normalize to [0, 1]
        image = image.to(device)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return

    # Model Inference
    model.eval()
    with torch.no_grad():
        try:
            pred, _ = model(image)
            res = pred[0]

            # Process the output
            res_slice = res[0, :, :, :].unsqueeze(0).cpu().detach().squeeze().numpy()
            res_slice = (res_slice - res_slice.min()) / (res_slice.max() - res_slice.min() + 1e-8)

            # Prepare the images for display
            cat_res = cv2.cvtColor((res_slice * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cat_ori = cv2.cvtColor(np.array(image[0].cpu().permute(1, 2, 0) * 255, dtype=np.uint8), cv2.COLOR_RGB2BGR)

            # Concatenate and save the result
            result = cv2.hconcat([cat_ori, cat_res])
            result_dir = os.path.join(work_dir, "result")
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            result_path = os.path.join(result_dir, image_name)
            cv2.imwrite(result_path, result)

            # Display result in Streamlit
            st.image(result, caption=f"Segmentation Result for {image_name}", use_container_width=True)
        except Exception as e:
            st.error(f"Error during model inference or image processing: {e}")

# Page Content Based on Sidebar Selection
if selected == "📹 Introduction":
    st.title("Guided Slot Attention for Unsupervised Video Object Segmentation")
    st.write("""
    This application demonstrates **Guided Slot Attention** for **Unsupervised Video Object Segmentation**. 
    Leveraging this method, we can detect and segment moving objects in videos without labeled data. 
    Explore our **Colab notebook** and **Streamlit app** to experience it firsthand!
    """)
   
    # Header
    st.header("Image Extractions with Different Strategies")

    # Load and display the main image
    image_path = "./assets/result.png"
    image = Image.open(image_path)
    st.image(image, caption="Result Overview", use_container_width=True)

    # Description for each part
    st.markdown("""
        - **A** : RGB Image
        - **B** : Optical Flow Map
        - **C** : Predicted Map
        - **D** : Ground Truth (GT)
        - **E** : Predicted Foreground RGB Slot Attention Map
        - **F** : Predicted Background RGB Slot Attention Map
        - **G** : Predicted Foreground Flow Slot Attention Map
        - **H** : Predicted Background Flow Slot Attention Map
    """)

    # Header
    st.header("Video Segmentation in Action")
    # Display the GIF with centered alignment and specified width
    gif_path = "./assets/GSANet.gif"
    st.image(gif_path, caption="Segmentation in Action", use_container_width=True)
    
    # Sample Segmentation Images
    st.header(" Segmentation training model metrics on Davis 2017")
    image1 = Image.open("assets/colab_run/davis-2017-result-1.png")
    image2 = Image.open("assets/colab_run/davis-2017-result-2.png")
    st.image([image1, image2], caption=["Result 1", "Result 2"])
    
    #  Segmentation Results on Davis 2017
    st.header("Segmentation Sample Results on Davis 2017")
    # Root directory where the folders are located
    root_dir = "log/2024-11-14 12:52:12-davis-2017/result/total"  # Replace with the path to the root directory

    # Loop through each folder in the root directory
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        
        # Check if the path is a directory
        if os.path.isdir(folder_path):
            st.header(folder_name.capitalize())  # Display the folder name as the title
            
            # Loop through each image file in the folder
            for image_file in os.listdir(folder_path):
                if image_file.endswith((".png", ".jpg", ".jpeg", ".gif")):  # Check for image file formats
                    image_path = os.path.join(folder_path, image_file)
                    
                    # Open and display the image
                    image = Image.open(image_path)
                    st.image(image, caption=image_file, use_container_width=True)   
    
    st.header("Segmentation Sample Results on Davis 2016")

    # Root directory where the folders are located
    root_dir = "log/davis-2016"  # Replace with the path to the root directory

    # Loop through each folder in the root directory
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        
        # Check if the path is a directory
        if os.path.isdir(folder_path):
            st.header(folder_name.capitalize())  # Display the folder name as the title
            
            # Counter to track the number of images displayed
            image_count = 0
            
            # Loop through each image file in the folder
            for image_file in os.listdir(folder_path):
                if image_file.endswith((".png", ".jpg", ".jpeg", ".gif")):  # Check for image file formats
                    image_path = os.path.join(folder_path, image_file)
                    
                    # Open and display the image
                    image = Image.open(image_path)
                    st.image(image, caption=image_file, use_container_width=True)
                    
                    # Increment the counter and break after displaying two images
                    image_count += 1
                    if image_count == 2:
                        break  # Stop after displaying 2 images in the folder
    

elif selected == "🛠️ Try It Out":
    st.title("Try Guided Slot Attention on Your Image/Video")
    st.write("Upload a image/video and adjust parameters to observe object segmentation.")
    
    # File uploader for images
    uploaded_images = st.file_uploader("Upload images for segmention", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_images:
        # uploaded_images a message while processing
        st.write("Processing your image...")
        for uploaded_image in uploaded_images:
            image = Image.open(uploaded_image)
            st.image(image, caption=uploaded_image.name, use_container_width=True)
            
            # Call the visual function with the image
            visual(image, uploaded_image.name)  # Pass the image and its name to the function

    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    frame_rate = st.slider("Select Frame Rate for Segmentation", min_value=1, max_value=30, value=10)
    if video_file:
        st.write("Processing your video with the selected frame rate...")
        # Placeholder for segmentation process
        st.video(video_file)

elif selected == "📊 Real-Life Applications":
    st.title("Real-Life Applications of Guided Slot Attention")
    st.write("""
    **Guided Slot Attention** can be applied across various fields:
    - **Autonomous Driving**: Improves object detection in dynamic environments.
    - **Surveillance Systems**: Enhances tracking and event detection.
    - **Medical Imaging**: Helps in tracking cells or anatomical structures across frames.
    """)

elif selected == "📜 FAQs":
    st.title("FAQs")
    st.write("""
    ### What is Guided Slot Attention?
    Guided Slot Attention is a framework for unsupervised object segmentation in video.
    
    ### How does it work?
    The model uses slot attention to learn features and improve segmentation iteratively.
    
    ### Can I upload my own videos?
    Yes! In the "Try It Out" section, upload a video to see segmentation results.
    """)

# Sidebar Footer: About and Contact
st.sidebar.header("About")
st.sidebar.write("""
**Guided Slot Attention App** is a demo of unsupervised video object segmentation. Created with Streamlit for interactive exploration.
""")
st.sidebar.write("⭐ [Star on GitHub](https://github.com/tan-nt/slot-attention-video-segmenter-app)")
st.sidebar.write("---")
st.sidebar.write("Developed by Your Name | Contact: [your.email@example.com](mailto:your.email@example.com)")

