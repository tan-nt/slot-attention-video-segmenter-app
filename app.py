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
    

# Load model function (you can adjust this based on your actual model setup)
def load_model():
    model = GSANet()  # Replace with actual model class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model).to(device)
    # Load model weights
    checkpoint = torch.load("log/2024-11-14 12:52:12-davis-2017/model/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

# Function to process image and remove background


# Function to process image and remove background with improved blending
def segment_and_replace_background(model, device, image, new_background):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust as per model requirements
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Prepare the input image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run the model
    with torch.no_grad():
        pred, _ = model(image_tensor)
        mask = pred[0].squeeze().cpu().numpy()  # Extract segmentation mask

    # Normalize and convert the mask to binary with smoother edges
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    binary_mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold and scale to 255
    
    # Resize binary mask to match the original image size
    binary_mask_resized = cv2.resize(binary_mask, image.size, interpolation=cv2.INTER_LINEAR)

    # Apply a larger Gaussian blur or a bilateral filter for smoother edges
    blurred_mask = cv2.GaussianBlur(binary_mask_resized, (25, 25), 0)  # Larger kernel for softer edges
    # Alternatively, use a bilateral filter to retain object edges better
    # blurred_mask = cv2.bilateralFilter(binary_mask_resized, d=15, sigmaColor=75, sigmaSpace=75)

    # Create a smooth alpha mask from the blurred mask
    mask_float = blurred_mask / 255.0  # Convert to [0,1] range for blending

    # Convert the original image to numpy and prepare the new background
    image_np = np.array(image)
    new_background = new_background.resize(image.size)
    new_background_np = np.array(new_background)

    # Apply smooth alpha blending to combine foreground and background
    foreground = (image_np * mask_float[..., None]).astype(np.uint8)  # Keep original colors of the foreground
    background = (new_background_np * (1 - mask_float[..., None])).astype(np.uint8)  # Apply inverted mask to background

    # Combine foreground and background with smooth blending
    combined_image = cv2.addWeighted(foreground, 1, background, 1, 0)
    return Image.fromarray(combined_image)

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
        
        
    # Streamlit Interface
    st.title("Real-life demo: Image Background Replacement with Guided Slot Attention Segmentation")

    # Image uploader
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Default background images
    default_backgrounds = {
        "Sunset": "assets/background/sunset.jpg",
        "Mountain": "assets/background/mountain.jpeg",
        "Sea": "assets/background/sea.jpg",
        "City": "assets/background/city.jpg"
    }

    # Select a default background or upload your own
    st.write("Choose a background:")
    background_choice = st.selectbox("Select a default background or upload your own:", ["Upload your own"] + list(default_backgrounds.keys()))
    new_background_image = None

    if background_choice == "Upload your own":
        # Handle user-uploaded background image
        new_background_image_file = st.file_uploader("Upload a background image", type=["jpg", "jpeg", "png"])
        if new_background_image_file is not None:
            # Open the uploaded image directly from the file uploader
            new_background_image = Image.open(new_background_image_file).convert("RGB")
    else:
        # Load the selected default background image using the file path
        new_background_image = Image.open(default_backgrounds[background_choice]).convert("RGB")

    if uploaded_image and new_background_image:
        # Display original image
        st.image(uploaded_image, caption="Original Image", use_column_width=True)

        # Load the uploaded image as a PIL Image object
        input_image = Image.open(uploaded_image).convert("RGB")

        # Load the model and setup device
        model, device = load_model()  # Assume load_model() is defined elsewhere

        # Process and display the result using the selected background
        result_image = segment_and_replace_background(model, device, input_image, new_background_image)
        st.image(result_image, caption="Image with New Background", use_column_width=True)

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

