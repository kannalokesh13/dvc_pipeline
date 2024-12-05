import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import sys 
from typing import List
import numpy as np


def load_or_build_engine(engine_path, onnx_model_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    if os.path.exists(engine_path):
        print("engine path",engine_path)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine:
                print("Engine loaded successfully.")
                return engine
            else:
                print("Failed to deserialize engine. Building new engine.")
    else:
        print("Engine file not found. Building new engine.")
    return build_engine(onnx_model_path, engine_path)

def build_engine(onnx_model_path, engine_path):
    # Use trtexec command or code to build engine
    os.system(f"trtexec --onnx={onnx_model_path} --saveEngine={engine_path}")
    return load_or_build_engine(engine_path, onnx_model_path)

def allocate_buffers(engine):
    """Allocates host and device buffer for TRT engine inference."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding)) 
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        
        # Host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings
        bindings.append(int(device_mem))
        
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """Run inference on a TensorRT engine."""
    # Transfer input data to the GPU
    [cuda.memcpy_htod_async(inp["device"], inp["host"], stream) for inp in inputs]
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU
    [cuda.memcpy_dtoh_async(out["host"], out["device"], stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    return [out["host"] for out in outputs]



def preprocess(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, ...]
    return img.astype(np.float32)


def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y 

def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y 

def postprocess(output, conf_threshold=0.5):
    # Assume output shape is (1, 84, 8400)
    output = output.squeeze(0)
    # Bounding boxes are the first 4 elements
    boxes = output[:4, :].transpose(1, 0)  
    obj_conf = output[4, :]
    # Extract class scores (80, 8400)
    class_scores = output[4:, :]
    confidence_threshold = 0.5  
    # class_scores is a NumPy array (shape: 80, 8400)
    max_class_scores = class_scores.max(axis=0)  # Shape: (8400,)
    class_indices = np.argmax(class_scores, axis=0)  # Shape: (8400,)
    # Filter boxes based on the confidence threshold
    high_conf_indices = max_class_scores > confidence_threshold
    filtered_boxes = boxes[high_conf_indices]
    filtered_scores = max_class_scores[high_conf_indices]
    filtered_classes = class_indices[high_conf_indices]
    indices = cv2.dnn.NMSBoxes(filtered_boxes, filtered_scores, score_threshold=0.5, nms_threshold=0.4)
    detections = []
    for bbox, score, label in zip(xywh2xyxy(filtered_boxes[indices]), filtered_scores[indices], filtered_classes[indices]):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,
                "class_name": label
            })
    return detections

def run_inference(engine_file_path, image_path,onnx_model_path):
    # Load the TensorRT engine
    engine = load_or_build_engine(engine_file_path,onnx_model_path)
    # Create execution context
    context = engine.create_execution_context()
    # Allocate buffers
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]
    input_size = (640, 640)
    scale_x = original_width / input_size[1]  # Scale factor for width
    scale_y = original_height / input_size[0]  # Scale factor for height
    print(img.shape)
    print(scale_x,scale_y)
    image = preprocess(img)  
    print(image.shape)
    input_name = engine.get_tensor_name(0)
    input_shape = engine.get_tensor_shape(input_name)[1:] 
    # Prepare input data (for example, reshape and normalize the input image)
    inputs[0]['host'] = image.ravel()
    # Perform inference
    trt_outputs = do_inference(context, bindings, inputs, outputs, stream)
    print(trt_outputs[0].shape)
    output = np.array(trt_outputs[0]).reshape(1, 84, 8400)
    detection = postprocess(output)
    for pred in detection:
        x1,y1,x2,y2 = pred['box'].astype(int)  # Convert box coordinates to integers
        class_name = str(pred['class_name'])
        confidence = pred['confidence']
        x1 = int(x1 * (original_width / 640))
        y1 = int(y1 * (original_height / 640))
        x2 = int(x2 * (original_width / 640))
        y2 = int(y2 * (original_height / 640))
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color in BGR format
        # Label with class name and confidence
        label = f'Class: {class_name}, Conf: {confidence:.2f}'
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    output_path = 'outputengine_v9.jpg'
    cv2.imwrite(output_path, img)
    return output_path


if __name__ == "__main__":
    # Get the filenames from command line arguments
    engine_file_path = sys.argv[1]
    image_path = sys.argv[2]
    onnx_model_path = sys.argv[3]
    # Call the inference function
    run_inference(engine_file_path, image_path, onnx_model_path)