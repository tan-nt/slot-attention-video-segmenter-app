import streamlit as st
import logging
from streamlit_option_menu import option_menu
from PIL import Image
import os
import os
import numpy as np
import cv2
import torchvision.transforms as transforms
import config as config
from PIL import Image
import tempfile

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


def visual_from_streamlit(device, model, images):
    model = model.to(device)
    model.eval()
    
    # Transform and normalize as required
    transform = transforms.Compose([
        transforms.Resize((config.TRAIN['img_size'], config.TRAIN['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = []
    with torch.no_grad():
        for image in images:
            image_name = image.name
            
            # Load and preprocess the image
            img_pil = Image.open(image).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            # Model prediction
            pred, _ = model(img_tensor)
            res = pred[0]
            
            # Process the prediction to display
            res = res.squeeze().cpu().numpy()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res_img = cv2.cvtColor((res * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            # Original image resized to match prediction size for side-by-side display
            ori_image = np.array(img_pil.resize((res_img.shape[1], res_img.shape[0])))
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR)
            
            # Concatenate original and prediction for comparison
            result = cv2.hconcat([ori_image, res_img])
            results.append((image_name, result))
    
    return results

def process_frame(device, model, frame):
    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((config.TRAIN['img_size'], config.TRAIN['img_size'])),  # Resize to model input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Convert the frame to PIL and apply transformations
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # Add batch dimension

    # Run the model prediction
    with torch.no_grad():
        pred, _ = model(img_tensor)
        res = pred[0]  # Assume the prediction output is compatible

    # Process the prediction result to match original image size and format
    res = res.squeeze().cpu().numpy()  # Remove unnecessary dimensions
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # Normalize to [0, 1]
    res_img = cv2.cvtColor((res * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # Convert to BGR for display

    # Resize original frame and prediction to original frame size if needed
    ori_image = np.array(img_pil.resize((res_img.shape[1], res_img.shape[0])))  # Resize back if necessary
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR)

    # Concatenate original and result for side-by-side comparison
    result_frame = cv2.hconcat([ori_image, res_img])

    return result_frame

# Process video and save each frame as an image
def process_video_to_images(temp_file_path, model, output_folder, frame_rate=20):
    cap = cv2.VideoCapture(temp_file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    frame_count = 0
    saved_images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames at intervals defined by frame_rate
        if frame_count % frame_rate == 0:
            processed_frame = process_frame(device, model, frame)
            image_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(image_path, processed_frame)
            saved_images.append(image_path)

        frame_count += 1

    cap.release()
    return saved_images


# Merge images into a video
def merge_images_to_video(image_paths, output_path, duration=0.5):
    img1 = cv2.imread(image_paths[0])
    height, width, _ = img1.shape
    frame_size = (width, height)

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_path, fourcc, 1, frame_size)

    for image_path in image_paths:
        img = cv2.imread(image_path)
        for _ in range(duration):
            video_writer.write(img)

    video_writer.release()

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GSANet()  # Initialize your model here
        model = torch.nn.DataParallel(model)
        
        # Assuming the model weights are loaded as before
        work_dir = "log/2024-11-14 12:52:12-davis-2017"
        model_dir = os.path.join(work_dir, "model")
        checkpoint = torch.load(model_dir + "/best_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        results = visual_from_streamlit(device, model, uploaded_images)
        
        # Display each result image
        for image_name, result in results:
            st.write(f"Image Name: {image_name}")
            st.image(result, caption="RGB Converted and Predicted Images", use_container_width=True)

        st.title("Video Segmentation with Frame Processing")

    # Upload video file and select frame rate
    video_file = st.file_uploader("Upload a video (Dataset exaple: https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)", type=["mp4", "mov", "avi"])
    frame_rate = st.slider("Select Frame Rate for Segmentation", min_value=20, max_value=100, value=50)
    
    # Use a persistent file location
    if video_file:
        st.write("Processing your video with the selected frame rate...")

        # Save the uploaded video to a persistent file location
        temp_file_path = "uploaded_video.mp4"
        with open(temp_file_path, "wb") as f:
            f.write(video_file.read())

        # Set up model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GSANet()  # Initialize your model here
        model = torch.nn.DataParallel(model)
        model.to(device)

        # Load model weights
        work_dir = "log/2024-11-14 12:52:12-davis-2017"
        model_dir = os.path.join(work_dir, "model")
        checkpoint = torch.load(model_dir + "/best_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Process the video and save each processed frame as an image
        cap = cv2.VideoCapture(temp_file_path)
        output_folder = "processed_frames"
        os.makedirs(output_folder, exist_ok=True)
        saved_images = []

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frames at intervals defined by frame_rate
            if frame_count % frame_rate == 0:
                processed_frame = process_frame(device, model, frame)
                image_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(image_path, processed_frame)
                saved_images.append(image_path)

            frame_count += 1

        cap.release()

        # Merge saved images into a final video
        output_video_path = "processed_video_output.mp4"
        merge_images_to_video(saved_images, output_video_path, duration=2)

        # Optionally delete the saved images after merging
        for image_path in saved_images:
            os.remove(image_path)

        # Display message to confirm saving
        st.write(f"Video processing complete. Saved as: {output_video_path}")

        # Display the processed video in Streamlit
        with open(output_video_path, 'rb') as video_file:
            st.video(video_file.read())
    
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

