from flask import Flask, request, redirect, url_for, render_template, send_file,flash,send_from_directory
import torch
import os
from ultralytics import YOLO
import zipfile
from uploading_version2 import process_data_folder
from downloading_version2 import process_output_folder
from config import list_available_versions, list_existing_versions, download_all_blobs
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug import Request
import shutil
import subprocess
from azure.storage.blob import BlobServiceClient
import os
import io
from onnx_conv import export_onnx,onnx_inference_v8
from config import list_available_versions
import onnx_conv_v9
from PIL import Image
import cv2
from pathlib import Path
from io import BytesIO
from yolov9 import YOLOv9
import uuid
from datetime import datetime
import csv


app = Flask(__name__)

Request.max_content_length = 5 * 1024 * 1024 * 1024  # 1 GB

Request.max_form_parts = 5 * 1024 * 1024 * 1024  # 1 GB

# Set the secret key for session management
app.secret_key = 'matdun123'

# Set up folders for uploading and converted files
UPLOAD_FOLDER = './'
CONVERTED_FOLDER = './'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONVERTED_FOLDER'] = CONVERTED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024

#Function for yolov9 onnx inference
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

#Function for yolov9 onnx inference

def inference_on_image(weights,source):
    print("[INFO] Intialize Model")
    detector = get_detector(weights,source)
    image = cv2.imread(source)

    print("[INFO] Inference Image")
    detections = detector.detect(image)
    detector.draw_detections(image, detections=detections)
    output_image_path = os.path.join('./','processed_' + os.path.basename(source))
    cv2.imwrite(output_image_path, image)
    return output_image_path

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
            print(f'Failed to delete {file_path}. Reason: {e}')


# Route to render upload page
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle the service selection and redirect
@app.route('/select_service', methods=['POST'])
def select_service():
    service = request.form.get('service')
    
    if service == 'weapon_service':
        return redirect(url_for('weapon_service'))
    else:
        # Handle other services if necessary
        return redirect(url_for('index'))

      
# Route to render the Weapon Service page
@app.route('/weapon_service')
def weapon_service():
    versions = list_available_versions()
    return render_template('weapon_service.html',versions = versions)

@app.route('/weapon_service/upload_data', methods=['GET', 'POST'])
def upload_folder():
    if request.method == 'POST':
        if 'data_folder' not in request.files:
            print("HERE")
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('data_folder')
        print(files)
        for file in files:
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            # Construct the file path within the upload folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            # Create necessary directories for nested folders
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Save the file
            file.save(file_path)
        
        print(next(os.walk(app.config['UPLOAD_FOLDER'])))
        actual_path = os.path.join(app.config['UPLOAD_FOLDER'],next(os.walk(app.config['UPLOAD_FOLDER']))[1][0])
        print(actual_path)
        # Process the files as needed (e.g., validation, etc.)
        local_upload_folder = None
        uploaded_result = process_data_folder(actual_path,local_upload_folder)
        print(uploaded_result)
        # Clean up the upload folder after processing
        cleanup_upload_folder(app.config['UPLOAD_FOLDER'])
        return render_template('weapon_service.html',versions = list_available_versions())
    return render_template('weapon_service.html')

def get_latest_train_folder(base_path="./runs/detect/"):
    all_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('train')]
    all_dirs.sort(key=lambda x: os.path.getctime(os.path.join(base_path, x)), reverse=True)
    
    if all_dirs:
        latest_folder = all_dirs[0]
        return os.path.join(base_path, latest_folder)
    return None

def add_data_to_csv(unique_id, created_time, location):

    csv_file_path = "./output_data.csv"
    if isinstance(created_time, (int, float)):  
        created_time = datetime.fromtimestamp(created_time)
    
    if isinstance(created_time, datetime):  # Ensure it's a datetime object
        created_time = created_time.strftime('%Y-%m-%d %H:%M:%S')
    

    file_exists = os.path.isfile(csv_file_path)
    
    
    data = [created_time, unique_id, location]
    
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
    
        if not file_exists:
            writer.writerow(['created_time', 'Id', 'location'])

        writer.writerow(data)

def zip_folder(folder_path, zip_stream):
    """Compress a folder into a zip archive."""
    with zipfile.ZipFile(zip_stream, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                zf.write(file_path, relative_path)

@app.route('/weapon_service/download_data',methods=['POST'])
def download_data():
    versions = list_available_versions()
    if request.method == 'POST':

        # version = request.form.get('version')
        # local_download_folder = './test/'
        # download_all_blobs(local_download_folder,version)
        # output_folder_path = './test/'
        target_folder = './test1/'
        # process_output_folder(output_folder_path, target_folder)


        artifacts_data_dir = './artifacts/data'
        os.makedirs(artifacts_data_dir, exist_ok=True)
        for item in os.listdir(target_folder):
            shutil.move(os.path.join(target_folder, item), artifacts_data_dir)


        
        versions = list_available_versions()
        return render_template('weapon_service.html',versions=versions)
    return render_template('weapon_service.html',versions = versions)
    
    

@app.route('/weapon_service/train', methods=['POST'])
def train():
    if request.method == 'POST':
        return render_template('train.html')
    
    return render_template('weapon_service.html')



@app.route('/weapon_service/training', methods=['POST'])
def training():
    if request.method == 'POST':

        model_file = request.form.get('model')
        epochs = request.form.get('epochs')
        image_size = request.form.get('imgs')
        model_file = model_file+".pt"
        # Load a model
        model = YOLO(model_file)  # load a pretrained model (recommended for training)
        # Train the model
        results = model.train(data="./artifacts/data/det_data.yaml", epochs=int(epochs), imgsz=image_size)  #change this one to artifacts dir
        # Path to the best weights (typically saved in the 'runs/train/exp/weights' folder)
        # Find the latest training folder
        latest_folder = get_latest_train_folder(base_path="./runs/detect/")
        if latest_folder:
            # Path to the best weights
            best_weights_path = os.path.join(latest_folder, "weights/best.pt")

            
            model_file = model_file.split(".")[0]
            artifacts_model_dir = f'./artifacts/models/yolo/{model_file}'
            os.makedirs(artifacts_model_dir, exist_ok=True)
            shutil.copy(best_weights_path, artifacts_model_dir)

            created_time = os.path.getctime(best_weights_path)
            unique_id = str(uuid.uuid4())
            add_data_to_csv(unique_id, created_time, best_weights_path)
            
            # Check if the best weights file exists
            if os.path.exists(best_weights_path):
                # Provide download option after training
                return '''
                    <h1>Model training completed!</h1>
                    <p>Click the button below to download the best weights (best.pt).</p>
                    <a href="/download_best_weights" download>
                        <button>Download best.pt</button>
                    </a>
                '''
            else:
                return "Training completed, but best.pt file not found!"
        else:
            return "No training folder found!"
        
@app.route('/download_best_weights')
def download_best_weights():
    latest_folder = get_latest_train_folder(base_path="./runs/detect/")
    
    if latest_folder:
        # Path to the best weights
        best_weights_path = os.path.join(latest_folder, "weights/best.pt")
        # Check if the best weights file exists
        if os.path.exists(best_weights_path):
            # Send the best.pt file for download
            return send_file(best_weights_path, as_attachment=True)
        else:
            return "No best.pt file found in the latest training folder!"
    else:
        return "No training folder found!"


@app.route('/weapon_service/convert', methods=['POST'])
def convert():
    if request.method == 'POST':
        return render_template('convert.html')
    
    return render_template('weapon_service.html')

# Route to handle file upload and conversion
@app.route('/weapon_service/convert_onnx', methods=['POST'])
def weapon_service_convert_onnx():
    if 'yolo_model' not in request.files:
        return redirect(request.url)
    file = request.files['yolo_model']
    if file.filename == '':
        return redirect(request.url)
    if file:
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(model_path)
        # Convert the YOLO model to ONNX
        onnx_model_path = export_onnx(model_path)
        print("model path",onnx_model_path)

        artifacts_onnx_dir = './artifacts/models/onnx_v8'
        os.makedirs(artifacts_onnx_dir, exist_ok=True)
        shutil.copy(onnx_model_path,artifacts_onnx_dir)

        # Render page with download button if conversion was successful
        return render_template('options_v8.html', onnx_file=onnx_model_path, conversion_success=True)

# Route to handle file upload and conversion
@app.route('/weapon_service/convert_onnx_v9', methods=['POST'])
def weapon_service_convert_onnx_v9():
    if 'yolo_model' not in request.files:
        return redirect(request.url)
    file = request.files['yolo_model']
    if file.filename == '':
        return redirect(request.url)
    if file:
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(model_path)
        # Convert the YOLO model to ONNX
        onnx_files = onnx_conv_v9.run(weights=model_path)
        if onnx_files:
            onnx_file = onnx_files[0]

            artifacts_onnx_dir = './artifacts/models/onnx_v9'
            os.makedirs(artifacts_onnx_dir, exist_ok=True)
            shutil.copy(onnx_file,artifacts_onnx_dir)

            # Render page with download button if conversion was successful
            return render_template('options_v9.html', onnx_file=onnx_file, conversion_success=True)


# Define route to handle inference
@app.route('/perform_inference_v8', methods=['POST'])
def perform_inference_v8():
    # Get the uploaded image and onnx file path from the form
    image_file = request.files['image']
    onnx_file = request.form['onnx_file']
    # Save the uploaded image temporarily
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)
    # Perform inference on the uploaded image
    result = onnx_inference_v8(onnx_file,image_path)
    output_image_path = "output_image.png"
    cv2.imwrite(output_image_path, result) 
    image_url = url_for('serve_file', filename=output_image_path)
    # Return the inference result (you can customize this part)
    return render_template('inference.html', image_url=image_url)

# Define route to handle inference
@app.route('/perform_inference', methods=['POST'])
def perform_inference():
    # Get the uploaded image and onnx file path from the form
    image_file = request.files['image']
    onnx_file = request.form['onnx_file']
    # Save the uploaded image temporarily
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)
    # Perform inference on the uploaded image
    result = inference_on_image(onnx_file,image_path)
    image_url = url_for('serve_file', filename=result)
    # Return the inference result (you can customize this part)
    return render_template('inference.html', image_url=image_url)


# Route to handle file upload and conversion
@app.route('/weapon_service/convert_tensorrt', methods=['POST'])
def weapon_service_convert_tensorrt():
    if 'yolo_model' not in request.files:
        return redirect(request.url)
    file = request.files['yolo_model']
    if file.filename == '':
        return redirect(request.url)
    if file:
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(model_path)
        # Trigger Docker container to convert the file
        onnx_path = export_onnx(model_path) 
        # Convert ONNX to TensorRT engine
        engine_file = os.path.splitext(file.filename)[0] + ".engine"
        engine_path = os.path.join(CONVERTED_FOLDER, engine_file)
        convert_onnx_to_tensorrt(onnx_path, engine_path)
        # Render page with download button if conversion was successful

        artifacts_tensorrt_dir = './artifacts/models/tensor_rt_v8'
        os.makedirs(artifacts_tensorrt_dir, exist_ok=True)
        shutil.copy(engine_path,artifacts_tensorrt_dir)

        return render_template('download_tensorrt.html', engine_file=engine_path, conversion_success=True)

# Route to handle file upload and conversion
@app.route('/weapon_service/convert_tensorrt_v9', methods=['POST'])
def weapon_service_convert_tensorrt_v9():
    if 'yolo_model' not in request.files:
        return redirect(request.url)
    file = request.files['yolo_model']
    if file.filename == '':
        return redirect(request.url)
    if file:
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(model_path)
        # Trigger Docker container to convert the file
        onnx_files = onnx_conv_v9.run(weights=model_path)
        if onnx_files:
            onnx_file = os.path.join(UPLOAD_FOLDER, os.path.basename(onnx_files[0])) 
            # Convert ONNX to TensorRT engine
            engine_file = os.path.splitext(file.filename)[0] + ".engine"
            engine_path = os.path.join(UPLOAD_FOLDER, engine_file)
            #convert_onnx_to_tensorrt(onnx_file, engine_path)
            # Render page with download button if conversion was successful

            artifacts_tensorrt_dir = './artifacts/models/tensor_rt_v9'
            os.makedirs(artifacts_tensorrt_dir, exist_ok=True)
            shutil.copy(engine_path,artifacts_tensorrt_dir)

            return render_template('download_tensorrt_v9.html', engine_file=engine_path, conversion_success=True)

# Define route to handle inference
@app.route('/perform_inference_trt_v8', methods=['POST'])
def perform_inference_trt_v8():
    # Get the uploaded image and onnx file path from the form
    current_dir = os.getcwd()
    image_file = request.files['image_file']
    if 'onnx_file' not in request.files:
        return "No ONNX file part in the request", 400
    onnx_file = request.files['onnx_file']
    engine_file = request.files['engine_file']
    # Save the uploaded image temporarily
    image_path =  image_file.filename#os.path.join('uploads', image_file.filename)
    onnx_file_path = onnx_file.filename #os.path.join('uploads', onnx_file.filename)
    engine_file_path = engine_file.filename #os.path.join('uploads', engine_file.filename)
    #engine_file.save(engine_file_path)
    onnx_file.save(onnx_file_path)
    image_file.save(image_path)
    docker_command = [
        "docker", "run", "--rm", "--gpus", "all", "-v", f"{current_dir}:/workspace",
        "tensorrt_inference_v8", "python3", "tensorrt_infer_trt.py",f"/workspace/{engine_file.filename}",
        f"/workspace/{image_file.filename}", f"/workspace/{onnx_file.filename}"
    ]
    # Execute the Docker command
    result = subprocess.run(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Perform inference on the uploaded image
    image_file = 'outputengine.jpg'
    image_url = url_for('serve_file', filename=image_file)
    # Return the inference result (you can customize this part)
    return render_template('inference.html', image_url=image_url)

@app.route('/perform_inference_trt_v9', methods=['POST'])
def perform_inference_trt_v9():
    # Get the uploaded image and onnx file path from the form
    current_dir = os.getcwd()
    image_file = request.files['image_file']
    if 'onnx_file' not in request.files:
        return "No ONNX file part in the request", 400
    onnx_file = request.files['onnx_file']
    engine_file = request.files['engine_file']
    # Save the uploaded image temporarily
    image_path =  image_file.filename#os.path.join('uploads', image_file.filename)
    onnx_file_path = onnx_file.filename #os.path.join('uploads', onnx_file.filename)
    engine_file_path = engine_file.filename #os.path.join('uploads', engine_file.filename)
    #engine_file.save(engine_file_path)
    onnx_file.save(onnx_file_path)
    image_file.save(image_path)
    docker_command = [
        "docker", "run", "--rm", "--gpus", "all", "-v", f"{current_dir}:/workspace",
        "tensorrt_inference_v9", "python3", "tensorrt_infer_trt_v9.py",f"/workspace/{engine_file.filename}",
        f"/workspace/{image_file.filename}", f"/workspace/{onnx_file.filename}"
    ]
    # Execute the Docker command
    result = subprocess.run(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Perform inference on the uploaded image
    image_file = 'outputengine_v9.jpg'
    image_url = url_for('serve_file', filename=image_file)
    # Return the inference result (you can customize this part)
    return render_template('inference.html', image_url=image_url)
      
# Function to run TensorRT conversion using Docker
def convert_onnx_to_tensorrt(onnx_file, engine_file):
    try:
        subprocess.run([
            'docker', 'run', '--rm',
            '--gpus', 'all',
            '-v', f'{os.path.abspath(CONVERTED_FOLDER)}:/workspace',
            'nvcr.io/nvidia/tensorrt:23.08-py3',  
            'trtexec', f'--onnx=/workspace/{os.path.basename(onnx_file)}',
            f'--saveEngine=/workspace/{os.path.basename(engine_file)}',
            '--explicitBatch'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during TensorRT conversion: {e}")
        raise


@app.route('/download/onnx/<path:onnx_file>')
def download_onnx(onnx_file):
    return send_file(onnx_file, as_attachment=True)

@app.route('/download/<path:engine_file>')
def download_file1(engine_file):
    return send_file(engine_file, as_attachment=True)


@app.route('/<filename>')
def serve_file(filename):
    return send_file(filename)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


if __name__ == '__main__':
    app.run(debug=True)