**Training pipeline for YOLOv8 and YOLOv9 models**

This is the training pipeline to 
- Upload data
- Download data
- Convert yolov8 model to onnx
- Convert yolov9 model to onnx
- Convert yolov8 model to tensorrt
- Convert yolov9 model to tensorrt
- Perform Yolov8 Onnx inference on image
- Perform Yolov9 Onnx inference on image
- Perform Yolov8 TensorRT inference on image
- Perform Yolov9 TensorRT inference on image

Run **python app.py** in the terminal for Flask server


There are three service weapon services, vehicle services and tracking services. select any service, but now only the weapon services are available and click on "Go"

![plot](./images/Screenshot%202024-12-09%20131446.png)

![plot](./images/Screenshot%202024-12-09%20131547.png)

## Upload 
To upload the dataset in YOLO format to Azure Blob Storage, select the desired folder containing the dataset and click the Upload button to initiate the process. Upon successful upload, the system will generate and provide a version number associated with the stored dataset. Once it has been uploaded, it will give a success message.

## Download
To download the dataset stored in Azure Blob Storage to the local system, select the desired version to be downloaded and click the Download button.

## Train Weapon Model 
To train the model using the downloaded dataset, click on the Train button. This will open a new interface where you need to select a pretrained model. The pretrained model section provides options for YOLOv8 and YOLOv9 models as listed below:

- YOLOv8

  - YOLOv8n
  - YOLOv8m
  - YOLOv8s
  - YOLOv8l
  - YOLOv8x

- YOLOv9

  - YOLOv9s
  - YOLOv9t
  - YOLOv9m
  - YOLOv9c
  - YOLOv9e

Next, specify the number of epochs for training and the desired image size for the model. Once all selections are made, click on the Start Train button to begin the training process. Once after training got completed, you can download the model.

![plot](./images/Screenshot%202024-12-09%20131611.png)

## Convert
To convert the PyTorch model into ONNX or TensorRT format, click on the Convert button. This will open a new interface where you can select different options for converting the model into ONNX or TensorRT formats.

- For YOLOv8 Models:

  - Select the desired model from the YOLOv8 section.
  - Click on the Convert to ONNX button to convert the PyTorch model to the ONNX format.
  - Once the conversion is complete, you can test the ONNX model by performing inference on a sample image.
  - Similarly, you can convert the model to the TensorRT format by selecting the Convert to TensorRT option.

- For YOLOv9 Models:

  Follow the same steps as for YOLOv8. Select the desired model from the YOLOv9 section, and choose the appropriate conversion option.
  Once the conversion process is completed, you can test the converted model by inferencing with a sample image.

![plot](./images/Screenshot%202024-12-09%20131627.png)


Run **streamlit run main.py** in the terminal for streamlit app
