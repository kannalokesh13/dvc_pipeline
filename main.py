import streamlit as st
import os
import shutil
from pathlib import Path
from uploading_version2 import process_data_folder
from downloading_version2 import process_output_folder
from config import list_available_versions, list_existing_versions, download_all_blobs
import subprocess
import torch
from ultralytics import YOLO
import zipfile
import shutil
import subprocess
import io
from onnx_conv import export_onnx,onnx_inference_v8
from config import list_available_versions
import onnx_conv_v9
from PIL import Image
import cv2
# Display the logo at the top of the app
st.set_page_config(
    page_title="YOLO Model",
    page_icon="./banner.jpg",  # Path to your favicon file
    layout="centered"
)
st.image("logo1.png", width=200)

# Set the title of the Streamlit app
st.title("Yolov8 and Yolov9 Training Model Pipeline")


# Set the upload folder path
UPLOAD_FOLDER = "'./'"
CONVERTED_FOLDER = "./"

from yolov9 import YOLOv9


def get_detector(weights,source):
    weights_path = weights
    classes_path = 'metadata.yaml'
    source_path = source
    assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
    assert os.path.isfile(classes_path), f"There's no classes file with name {weights_path}"
    assert os.path.isfile(source_path), f"There's no source file with name {weights_path}"


    image = cv2.imread(source_path)
    h,w = image.shape[:2]

    detector = YOLOv9(model_path=weights_path,
                      class_mapping_path=classes_path,
                      original_size=(w, h),
                      score_threshold=0.5,
                      conf_thresold=0.5,
                      iou_threshold=0.4,
                      device='cpu')
    return detector

def inference_on_image(weights,source):
    print("[INFO] Intialize Model")
    detector = get_detector(weights,source)
    image = cv2.imread(source)

    print("[INFO] Inference Image")
    detections = detector.detect(image)
    detector.draw_detections(image, detections=detections)
    output_image_path = os.path.join('./','processed_' + os.path.basename(source))
    cv2.imwrite(output_image_path, image)

    #output_path = f"output/{Path(args.source).name}"
    #print(f"[INFO] Saving result on {output_path}")
    return output_image_path

def get_latest_train_folder(base_path="./runs/detect/"):
    # Function to find the latest training folder
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    if not folders:
        return None
    latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(base_path, f)))
    return os.path.join(base_path, latest_folder)

def train_model(model_file, epochs, image_size):
    # Load a model
    model = YOLO(model_file)  # load a pretrained model (recommended for training)
    
    # Train the model
    results = model.train(data="./test1/det_data.yaml", epochs=int(epochs), imgsz=int(image_size))
    
    # Find the latest training folder
    latest_folder = get_latest_train_folder(base_path="./runs/detect/")
    return latest_folder



def convert_last_trained_model():
    # Logic to convert the last trained model to ONNX
    latest_folder = get_latest_train_folder(base_path="./runs/detect/")
    if latest_folder:
        best_weights_path = os.path.join(latest_folder, "weights/best.pt")
        if os.path.exists(best_weights_path):
            print(best_weights_path)
            return export_onnx(best_weights_path)
    return None

def convert_last_trained_v9_model():
    # Logic to convert the last trained model to ONNX
    latest_folder = get_latest_train_folder(base_path="./runs/detect/")
    if latest_folder:
        best_weights_path = os.path.join(latest_folder, "weights/best.pt")
        if os.path.exists(best_weights_path):
            onnx_path = onnx_conv_v9.run(weights=best_weights_path)
            return onnx_path[0]
    return None

def convert_last_trained_model_trt():
    # Logic to convert the last trained model to ONNX
    latest_folder = get_latest_train_folder(base_path="./runs/detect/")
    if latest_folder:
        best_weights_path = os.path.join(latest_folder, "weights/best.pt")
        if os.path.exists(best_weights_path):
            print(best_weights_path)
            onnx_path= export_onnx(best_weights_path)
            return onnx_path, convert_onnx_to_tensorrt(onnx_path, engine_file)
    return None

def convert_last_trained_v9_model_trt():
    # Logic to convert the last trained model to ONNX
    latest_folder = get_latest_train_folder(base_path="./runs/detect/")
    if latest_folder:
        best_weights_path = os.path.join(latest_folder, "weights/best.pt")
        if os.path.exists(best_weights_path):
            onnx_path = onnx_conv_v9.run(weights=best_weights_path)
            return onnx_path[0], convert_onnx_to_tensorrt1(onnx_path[0], engine_file)
    return None

def filter_output(output):
    """Filter out unwanted verbose output."""
    filtered_lines = []
    for lines in output.splitlines():
        if "size" in lines or "AugmentOp" in lines:
            continue  # Skip lines containing specific keywords (e.g., 'size', 'AugmentOp')
        success_messages = [line for line in lines if "Validate Epoch" in line or "Train Epoch" in line and "loss=" in line]
        if success_messages:continue
        filtered_lines.append(lines)
    return "\n".join(filtered_lines)

# Ensure the upload folder exists
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Function to run TensorRT conversion using Docker
def convert_onnx_to_tensorrt(onnx_file, engine_file):
    try:
        subprocess.run([
            'docker', 'run', '--rm',
            '--gpus', 'all',
            '-v', f'{os.path.abspath(CONVERTED_FOLDER)}:/workspace/',
            'nvcr.io/nvidia/tensorrt:23.08-py3',  
            'trtexec', f'--onnx=/workspace/{os.path.basename(onnx_file)}',
            f'--saveEngine=/workspace/{os.path.basename(engine_file)}',
            '--explicitBatch'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during TensorRT conversion: {e}")
        raise
# Function to run TensorRT conversion using Docker
def convert_onnx_to_tensorrt1(onnx_file, engine_file):
    try:
        subprocess.run([
            'docker', 'run', '--rm',
            '--gpus', 'all',
            '-v', f'{os.path.abspath(CONVERTED_FOLDER)}:/workspace/',
            'nvcr.io/nvidia/tensorrt:23.08-py3',  
            'trtexec', f'--onnx=/workspace/uploads/{os.path.basename(onnx_file)}',
            f'--saveEngine=/workspace/uploads/{os.path.basename(engine_file)}',
            '--explicitBatch'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during TensorRT conversion: {e}")
        raise

def cleanup_upload_folder(folder):
    """Delete the contents of the upload folder."""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory and its contents
        except Exception as e:
            st.error(f'Failed to delete {file_path}. Reason: {e}')

# Section for uploading data for training
st.header("Upload Data for Training")
uploaded_files = st.file_uploader("Upload your data files", accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files to the server (streamlit folder)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        # Create directories if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Get actual folder path of the upload directory
    actual_path = UPLOAD_FOLDER
    st.success(f"Files successfully uploaded to {actual_path}")

    # Process the uploaded folder
    if st.button("Process Uploaded Folder"):
        uploaded_result = process_data_folder(actual_path)
        st.info(uploaded_result)

    # Clean up the upload folder
    if st.button("Clean Up Upload Folder"):
        cleanup_upload_folder(UPLOAD_FOLDER)
        st.success("Upload folder cleaned up.")

else:
    st.info("Please upload some data files.")
# Section for downloading data
# Streamlit Layout for Download Functionality
st.title("Download Data")

# Step 1: List Available Versions for Download
st.header("Choose Version for Download")
versions = list_available_versions()  # Fetch available versions
selected_version = st.selectbox("Select a version", versions)

# Step 2: Trigger Download and Process the Folder
if st.button("Download"):
    if selected_version:
        local_download_folder = './azure_downloads'
        target_folder = './data_download'

        # Simulate downloading blobs from Azure
        download_all_blobs(local_download_folder, selected_version)

        # Process the downloaded folder
        process_output_folder(local_download_folder, target_folder)

        # Refresh version list after processing (if needed)
        versions = list_available_versions()
        st.success("Folder Downloaded Successfully.")

    else:
        st.error("Please select a version to download.")

# Streamlit layout for training the model


st.title(" Train YOLOv8 and v9 Model")

#model_file = st.text_input('Enter the model file name (without .pt extension):', 'yolov8')
model_file = st.selectbox("Select YOLO Model", ["yolov8n", "yolov8s", "yolov8m", "yolov9s","yolov9t","yolov9c","yolov9e"])
epochs = st.number_input('Enter the number of epochs:', min_value=1, max_value=100, value=10, step=1)
image_size = st.number_input('Enter the image size:', min_value=32, max_value=1024, value=640, step=32)
    
if st.button('Start Training'):
    
    model_file = model_file + ".pt"
    
    st.write("Model:", model_file)
    st.write("Epochs:", epochs)
    st.write("Image Size:", image_size)

    # Call the training function
    with st.spinner('Training in progress...'):
        
        latest_folder = train_model(model_file, epochs, image_size)
    
    if latest_folder:
        best_weights_path = os.path.join(latest_folder, "weights/best.pt")
        if os.path.exists(best_weights_path):
            st.success("Model training completed!")
            st.write("Best model saved at:", best_weights_path)
            
            # Provide download link for best.pt
            with open(best_weights_path, "rb") as file:
                st.download_button(
                    label="Download best.pt",
                    data=file,
                    file_name="best.pt",
                    mime="application/octet-stream"
                )
        else:
            st.error("Training completed, but best.pt file not found!")
    else:
        st.error("No training folder found!")
    

st.title("YOLO Model ONNX Conversion Service")

# Initialize session state variables if they don't already exist
if 'onnx_conversion_status' not in st.session_state:
    st.session_state.conversion_status = {
        'yolov8_upload_onnx': False,'yolov8_trained_onnx': False,
        'yolov9_upload_onnx': False,'yolov9_trained_onnx': False,
        'yolov8_trained_onnx1': False,'yolov9_trained_onnx1': False,
    }

if 'uploaded_model_path' not in st.session_state:
    st.session_state.uploaded_model_path = ""

if 'onnx_model_path' not in st.session_state:
    st.session_state.onnx_model_path = ""

option = st.radio(
    "Choose an option:",
    ('Upload YOLOv8 Model for ONNX Conversion', 'Upload YOLOv9 Model for ONNX Conversion', 
     'Convert Last Trained YOLOv8 Model to ONNX', 'Convert Last Trained YOLOv9 Model to ONNX'
     ,'YOLO v8 Model Inference on converted ONNX','YOLO v9 Model Inference on converted ONNX')
)

# Option 1: Upload a model to convert to ONNX
if option == "Upload YOLOv8 Model for ONNX Conversion":
    uploaded_file = st.file_uploader("Upload a YOLOv8 model file (.pt)", type=['pt'])
    if uploaded_file is not None:
        # Save the uploaded model file
        st.session_state.uploaded_model_path = os.path.join("uploads", uploaded_file.name)
        with open(st.session_state.uploaded_model_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.conversion_status['yolov8_upload_onnx'] = True
        st.success(f"Uploaded YOLOv8 model: {uploaded_file.name}")
        if st.session_state.conversion_status['yolov8_upload_onnx'] and not st.session_state.conversion_status['yolov8_trained_onnx']:
                if st.button("Convert to ONNX"):
                    with st.spinner('Converting to ONNX...'):
                        st.session_state.onnx_model_path = export_onnx(st.session_state.uploaded_model_path)
                        if st.session_state.onnx_model_path:
                        
                            st.session_state.conversion_status['yolov8_trained_onnx'] = True
                            st.success(f"Model converted to Onnx: {st.session_state.onnx_model_path}")
                            with open(st.session_state.onnx_model_path, "rb") as f:
                                st.download_button("Download ONNX model", f, file_name=st.session_state.onnx_model_path)

elif option == "Upload YOLOv9 Model for ONNX Conversion":
    uploaded_file = st.file_uploader("Upload a YOLOv9 model file (.pt)", type=['pt'])
    if uploaded_file is not None:
        # Save the uploaded model file
        st.session_state.uploaded_model_path = os.path.join("uploads", uploaded_file.name)
        with open(st.session_state.uploaded_model_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.conversion_status['yolov9_upload_onnx'] = True
        st.success(f"Uploaded YOLOv9 model: {uploaded_file.name}")
        if st.session_state.conversion_status['yolov9_upload_onnx'] and not st.session_state.conversion_status['yolov9_trained_onnx']:
                if st.button("Convert to ONNX"):
                    with st.spinner('Converting to ONNX...'):
                        st.session_state.onnx_model = onnx_conv_v9.run(weights=st.session_state.uploaded_model_path)
                        st.session_state.onnx_model_path = st.session_state.onnx_model[0]
                        if st.session_state.onnx_model_path:
                            st.session_state.conversion_status['yolov9_trained_onnx'] = True
                            st.success(f"Model converted to ONNX: {st.session_state.onnx_model_path}")
                            with open(st.session_state.onnx_model_path, "rb") as f:
                                st.download_button("Download ONNX model", f, file_name=st.session_state.onnx_model_path)

# Option 3: Convert the last trained model to ONNX
elif option == "Convert Last Trained YOLOv8 Model to ONNX":
    # Check if conversion has already been performed
    if not st.session_state.conversion_status.get('yolov8_trained_onnx1'):
        if st.button("Convert to ONNX"):
            with st.spinner('Converting to ONNX...'):
                st.write("Converting the last trained YOLOv8 model to ONNX...")
                
                # Convert the last trained model to ONNX and copy to uploads folder
                st.session_state.onnx_model_path = convert_last_trained_model()
                
                if st.session_state.onnx_model_path:
                    shutil.copy(st.session_state.onnx_model_path, './uploads/')
                    st.session_state.uploaded_model_path = os.path.join("uploads", os.path.basename(st.session_state.onnx_model_path))
                    st.write(f"Uploaded model path: {st.session_state.uploaded_model_path}")
                   
                    st.session_state.conversion_status['yolov8_trained_onnx1'] = True
                    st.success(f"Model converted to ONNX: {st.session_state.onnx_model_path}")
                    with open(st.session_state.onnx_model_path, "rb") as f:
                        st.download_button("Download ONNX model", f, file_name=st.session_state.onnx_model_path)   


# Option 4: Convert the last trained model to ONNX
elif option == "Convert Last Trained YOLOv9 Model to ONNX":

    # Check if conversion has already been performed
    if not st.session_state.conversion_status.get('yolov9_trained_onnx1'):
        if st.button("Convert to ONNX"):
            with st.spinner('Converting to ONNX...'):
                st.write("Converting the last trained YOLOv9 model to ONNX...")
                
                # Convert the last trained model to ONNX and copy to uploads folder
                st.session_state.onnx_model_path = convert_last_trained_v9_model()
                
                if st.session_state.onnx_model_path:
                    shutil.copy(st.session_state.onnx_model_path, './uploads/')
                    st.session_state.uploaded_model_path = os.path.join("uploads", os.path.basename(st.session_state.onnx_model_path))
                    st.write(f"Uploaded model path: {st.session_state.uploaded_model_path}")
                    
                    st.session_state.conversion_status['yolov9_trained_onnx1'] = True
                    st.success(f"Model converted to TensorRT: {st.session_state.onnx_model_path}")
                    with open(st.session_state.onnx_model_path, "rb") as f:
                        st.download_button("Download ONNX model", f, file_name=st.session_state.onnx_model_path)   

elif option == "YOLO v8 Model Inference on converted ONNX": 
        # Display file uploader immediately
        uploaded_file = st.file_uploader("Upload an image", type=['jpg'])
        # Only show the inference button after an image is uploaded
        if uploaded_file is not None:
            uploaded_image_path = os.path.join("uploads", uploaded_file.name)
            with open(uploaded_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Image uploaded: {uploaded_file.name}")

            # Show the inference button after the image is uploaded
            if st.button("Perform ONNX inference"):
                with st.spinner('Running Inference on ONNX...'):
                    st.write("Performing inference on the last trained YOLOv8 model with ONNX...")
                    st.session_state.onnx_model_path = convert_last_trained_model()
                    # Path to the ONNX model
                    onnx_model_path = st.session_state.onnx_model_path
                    st.write(onnx_model_path)
                    results = onnx_inference_v8(onnx_model_path,uploaded_image_path)
                    results_rgb = cv2.cvtColor(results, cv2.COLOR_BGR2RGB)
                    st.image(results_rgb, caption="Inference Result")
        else:
            st.warning("Please upload an image to perform inference.")
    
elif option == "YOLO v9 Model Inference on converted ONNX":
        # Display file uploader immediately
        uploaded_file = st.file_uploader("Upload an image", type=['jpg'])
        # Only show the inference button after an image is uploaded
        if uploaded_file is not None:
            uploaded_image_path = os.path.join("uploads", uploaded_file.name)
            with open(uploaded_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Image uploaded: {uploaded_file.name}")

            # Show the inference button after the image is uploaded
            if st.button("Perform ONNX inference"):
                with st.spinner('Running Inference on ONNX...'):
                    st.write("Performing inference on the last trained YOLOv9 model with ONNX...")
                    st.session_state.onnx_model_path = convert_last_trained_v9_model()
                    # Path to the ONNX model
                    onnx_model_path = st.session_state.onnx_model_path
                    st.write(onnx_model_path)
                    results = inference_on_image(onnx_model_path,uploaded_image_path)
                    st.image(results, caption="Inference Result")
        else:
            st.warning("Please upload an image to perform inference.")
    

st.title("YOLO Model TensorRT Conversion Service")


# Initialize session state variables if they don't already exist
if 'trt_conversion_status' not in st.session_state:
    st.session_state.trt_conversion_status = {
        'yolov8_upload_trt': False,'yolov8_trained_trt': False,
        'yolov9_upload_trt': False,'yolov9_trained_trt': False,
        'yolov8_trained_trt1': False,'yolov9_trained_trt1': False,
    }

if 'uploaded_model_path' not in st.session_state:
    st.session_state.uploaded_model_path = ""

if 'onnx_model_path' not in st.session_state:
    st.session_state.onnx_model_path = ""

option = st.radio(
    "Choose an option:",
    ('Upload YOLOv8 Model for TensorRT Conversion', 'Upload YOLOv9 Model for TensorRT Conversion', 
     'Convert Last Trained YOLOv8 Model to TensorRT', 'Convert Last Trained YOLOv9 Model to TensorRT',
     'YOLO v8 Model Inference on converted TensorRT','YOLO v9 Model Inference on converted TensorRT'
     )
)
# Option 1: Upload a model to convert to ONNX
if option == "Upload YOLOv8 Model for TensorRT Conversion":
    uploaded_file = st.file_uploader("Upload YOLOv8 Model for TensorRT Conversion (.pt)", type=['pt'])
    
    if uploaded_file is not None:
        # Save the uploaded model file
        st.session_state.uploaded_model_path = os.path.join("uploads", uploaded_file.name)
        with open(st.session_state.uploaded_model_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.trt_conversion_status['yolov8_upload_trt'] = True
        st.success(f"Uploaded YOLOv8 model: {uploaded_file.name}")
        if st.session_state.trt_conversion_status['yolov8_upload_trt'] and not st.session_state.trt_conversion_status['yolov8_trained_trt']:
                if st.button("Convert to TensorRT"):
                    with st.spinner('Converting to TensorRT...'):
                        st.session_state.onnx_model_path = export_onnx(st.session_state.uploaded_model_path)
                        if st.session_state.onnx_model_path:
                            engine_path = os.path.splitext(st.session_state.onnx_model_path)[0] + ".engine"
                            convert_onnx_to_tensorrt(st.session_state.onnx_model_path, engine_path)
                            st.session_state.trt_conversion_status['yolov8_trained_trt'] = True
                            st.success(f"Model converted to TensorRT: {engine_path}")
                            with open(engine_path, "rb") as f:
                                st.download_button("Download TensorRT model", f, file_name=engine_path)

elif option == "Upload YOLOv9 Model for TensorRT Conversion":
    uploaded_file = st.file_uploader("Upload YOLOv9 Model for TensorRT Conversion(.pt)", type=['pt'])
    if uploaded_file is not None:
        # Save the uploaded model file
        st.session_state.uploaded_model_path = os.path.join("uploads", uploaded_file.name)
        with open(st.session_state.uploaded_model_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.trt_conversion_status['yolov9_upload_trt'] = True
        st.success(f"Uploaded YOLOv9 model: {uploaded_file.name}")
        if st.session_state.trt_conversion_status['yolov9_upload_trt'] and not st.session_state.trt_conversion_status['yolov9_trained_trt']:
                if st.button("Convert to TensorRT"):
                    with st.spinner('Converting to TensorRT...'):
                        st.session_state.onnx_model = onnx_conv_v9.run(weights=st.session_state.uploaded_model_path)
                        st.session_state.onnx_model_path = st.session_state.onnx_model[0]
                        if st.session_state.onnx_model_path:
                            engine_path = os.path.splitext(st.session_state.onnx_model_path)[0] + ".engine"
                            convert_onnx_to_tensorrt1(st.session_state.onnx_model_path, engine_path)
                            st.session_state.trt_conversion_status['yolov9_trained_trt'] = True
                            st.success(f"Model converted to TensorRT: {engine_path}")
                            with open(engine_path, "rb") as f:
                                st.download_button("Download TensorRT model", f, file_name=engine_path)

# Option 3: Convert the last trained model to ONNX
elif option == "Convert Last Trained YOLOv8 Model to TensorRT":
    # Check if conversion has already been performed
    if not st.session_state.trt_conversion_status.get('yolov8_trained_trt1'):
        if st.button("Convert to TensorRT"):
            with st.spinner('Converting to TensorRT...'):
                st.write("Converting the last trained YOLOv8 model to TensorRT...")
                
                # Convert the last trained model to ONNX and copy to uploads folder
                st.session_state.onnx_model_path = convert_last_trained_model()
                
                if st.session_state.onnx_model_path:
                    shutil.copy(st.session_state.onnx_model_path, './uploads/')
                    st.session_state.uploaded_model_path = os.path.join("uploads", os.path.basename(st.session_state.onnx_model_path))
                    st.write(f"Uploaded model path: {st.session_state.uploaded_model_path}")
                    engine_file = st.session_state.uploaded_model_path.split('/')[-1].split('.')[0] + ".engine"
                    print("engine",engine_file)
                    convert_onnx_to_tensorrt(st.session_state.uploaded_model_path, engine_file)
                    st.session_state.trt_conversion_status['yolov8_trained_trt1'] = True
                    st.success(f"Model converted to TensorRT: {engine_file}")
                    shutil.copy(engine_file, './uploads/')
                    with open(engine_file, "rb") as f:
                        st.download_button("Download TensorRT model", f, file_name=engine_file)   


elif option == "Convert Last Trained YOLOv9 Model to TensorRT":

    # Check if conversion has already been performed
    if not st.session_state.trt_conversion_status.get('yolov9_trained_trt1'):
        if st.button("Convert to TensorRT"):
            with st.spinner('Converting to TensorRT...'):
                st.write("Converting the last trained YOLOv9 model to TensorRT...")
                
                # Convert the last trained model to ONNX and copy to uploads folder
                st.session_state.onnx_model_path = convert_last_trained_v9_model()
                
                if st.session_state.onnx_model_path:
                    shutil.copy(st.session_state.onnx_model_path, './uploads/')
                    st.session_state.uploaded_model_path = os.path.join("uploads", os.path.basename(st.session_state.onnx_model_path))
                    st.write(f"Uploaded model path: {st.session_state.uploaded_model_path}")
                    engine_file = os.path.splitext(st.session_state.uploaded_model_path)[0] + ".engine"
                    convert_onnx_to_tensorrt1(st.session_state.uploaded_model_path, engine_file)
                    st.session_state.trt_conversion_status['yolov9_trained_trt1'] = True
                    st.success(f"Model converted to TensorRT: {engine_file}")
                    with open(engine_file, "rb") as f:
                        st.download_button("Download TensorRT model", f, file_name=engine_file)   

elif option == "YOLO v8 Model Inference on converted TensorRT": 
        # Display file uploader immediately
        uploaded_file = st.file_uploader("Upload an image", type=['jpg'])
        # Only show the inference button after an image is uploaded
        if uploaded_file is not None:
            uploaded_image_path = os.path.join("uploads", uploaded_file.name)
            with open(uploaded_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Image uploaded: {uploaded_file.name}")
            # Show the inference button after the image is uploaded
            if st.session_state.onnx_model_path in st.session_state or 'yolov8_trained_trt1' in st.session_state.trt_conversion_status:
                engine_model_path = os.path.splitext(st.session_state.onnx_model_path)[0] + ".engine"
                st.write(f"Using saved ONNX and TensorRT models for inference: {engine_model_path}")

                # Show the inference button after the image is uploaded
                if st.button("Perform TensorRT inference"):
                    with st.spinner('Running Inference on TensorRT...'):
                        st.write("Performing inference using the last trained YOLOv8 model with TensorRT...")
                        current_dir = os.getcwd()
                        # Prepare the Docker command for inference
                        docker_command = [
                            "docker", "run", "--rm", "--gpus", "all", "-v", f"{current_dir}:/workspace",
                            "tensorrt_inference_v8", "python3", "tensorrt_infer_trt.py",
                            f"/workspace/{engine_model_path}", f"/workspace/{uploaded_image_path}", 
                            f"/workspace/{st.session_state.onnx_model_path}"
                        ]
                        # Execute the Docker command
                        result = subprocess.run(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        # Assuming the output image is saved as 'outputengine.jpg'
                        image_file = 'outputengine.jpg'
                        st.image(image_file, caption="Inference Result")
            else:
                st.warning("Please first convert the YOLOv8 model to TensorRT.")
        
elif option == "YOLO v9 Model Inference on converted TensorRT": 
        # Display file uploader immediately
        uploaded_file = st.file_uploader("Upload an image", type=['jpg'])
        # Only show the inference button after an image is uploaded
        if uploaded_file is not None:
            uploaded_image_path = os.path.join("uploads", uploaded_file.name)
            with open(uploaded_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Image uploaded: {uploaded_file.name}")
            # Show the inference button after the image is uploaded
            if st.session_state.onnx_model_path in st.session_state or 'yolov9_trained_trt1' in st.session_state.trt_conversion_status:
                engine_model_path = os.path.splitext(st.session_state.onnx_model_path)[0] + ".engine"
                st.write(f"Using saved ONNX and TensorRT models for inference: {engine_model_path}")

                # Show the inference button after the image is uploaded
                if st.button("Perform TensorRT inference"):
                    with st.spinner('Running Inference on TensorRT...'):
                        st.write("Performing inference using the last trained YOLOv9 model with TensorRT...")
                        current_dir = os.getcwd()
                        # Prepare the Docker command for inference
                        docker_command = [
                            "docker", "run", "--rm", "--gpus", "all", "-v", f"{current_dir}:/workspace",
                            "tensorrt_inference_v9", "python3", "tensorrt_infer_trt_v9.py",
                            f"/workspace/{engine_model_path}", f"/workspace/{uploaded_image_path}", 
                            f"/workspace/{st.session_state.onnx_model_path}"
                        ]
                        # Execute the Docker command
                        result = subprocess.run(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        # Assuming the output image is saved as 'outputengine.jpg'
                        image_file = 'outputengine_v9.jpg'
                        st.image(image_file, caption="Inference Result")
            else:
                st.warning("Please first convert the YOLOv9 model to TensorRT.")
