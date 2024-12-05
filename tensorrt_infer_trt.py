import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import sys 

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
    input_name = engine.get_tensor_name(0)
    boxes_name = engine.get_tensor_name(1)
    scores_name = engine.get_tensor_name(2)
    classes_name = engine.get_tensor_name(3)  
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape(input_name)), dtype=np.float32)
    h_output_boxes = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape(boxes_name)), dtype=np.float32)
    h_output_scores = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape(scores_name)), dtype=np.float32)
    h_output_classes = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape(classes_name)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output_boxes = cuda.mem_alloc(h_output_boxes.nbytes)
    d_output_scores = cuda.mem_alloc(h_output_scores.nbytes)
    d_output_classes = cuda.mem_alloc(h_output_classes.nbytes)
    stream = cuda.Stream()
    return h_input, h_output_boxes, h_output_scores, h_output_classes, d_input, d_output_boxes, d_output_scores, d_output_classes, stream



def do_inference(context, h_input, h_output, d_input, d_output, stream, image):
    # Transfer input data to the GPU
    cuda.memcpy_htod_async(d_input, image, stream)
    # Run inference
    context.execute_async(batch_size=1, bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
    return h_output

def preprocess(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, ...]
    return img.astype(np.float32)


def post_process(original_img, boxes, scores, classes):
    print("original_image",original_img.shape)
    img = cv2.resize(original_img, (640, 640))
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    conf_thres = 0.25
    iou_thres = 0.45
    index = cv2.dnn.NMSBoxes(boxes.tolist(), scores.flatten().tolist(), conf_thres, iou_thres)
    for bbox, cls, score in zip(boxes[index], classes[index], scores[index]):
        label = f'{int(cls)}, {score:.2f}'
        xywh = [int(i) for i in bbox]
        cv2.rectangle(img, (xywh[0], xywh[1], xywh[2], xywh[3]), (0, 255, 0), 2)
        cv2.putText(img, label, (xywh[0], xywh[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img


def run_tensorrt_inference(engine_file_path, image_path,onnx_model_path):
    # Load TensorRT engine
    engine = load_or_build_engine(engine_file_path,onnx_model_path)
    h_input, h_boxes, h_scores, h_classes, d_input, d_boxes, d_scores, d_classes, stream = allocate_buffers(engine)
    context = engine.create_execution_context()
    # Use get_tensor_name to get the input tensor name
    input_name = engine.get_tensor_name(0)
    # Get the input shape using the tensor name
    input_shape = engine.get_tensor_shape(input_name)[1:] 
    context = engine.create_execution_context()
    # Use get_tensor_name to get the input tensor name
    input_name = engine.get_tensor_name(0)
    img = cv2.imread(image_path)
    image = preprocess(img)  
    np.copyto(h_input, image.ravel())  # Copy the image data into the input buffer
    # Transfer input to the GPU
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Execute the model
    context.execute_async_v2(bindings=[int(d_input), int(d_boxes), int(d_scores), int(d_classes)], stream_handle=stream.handle)
    # Transfer output back to the host
    cuda.memcpy_dtoh_async(h_boxes, d_boxes, stream)
    cuda.memcpy_dtoh_async(h_scores, d_scores, stream)
    cuda.memcpy_dtoh_async(h_classes, d_classes, stream)
    # Synchronize the stream
    stream.synchronize()
    # Post-process the output to extract results
    boxes = h_boxes[:trt.volume(engine.get_tensor_shape(engine.get_tensor_name(1)))]  
    scores = h_scores[:trt.volume(engine.get_tensor_shape(engine.get_tensor_name(2)))]
    classes = h_classes[:trt.volume(engine.get_tensor_shape(engine.get_tensor_name(3)))]
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    classes = classes.reshape(-1)
    output_img = post_process(img, boxes, scores, classes)
    output_path = 'outputengine.jpg'
    cv2.imwrite(output_path, output_img)
    return output_path


   
    #return h_input, h_boxes, h_scores, h_classes, d_input, d_boxes, d_scores, d_classes, stream

if __name__ == "__main__":
    # Get the filenames from command line arguments
    engine_file_path = sys.argv[1]
    image_path = sys.argv[2]
    onnx_model_path = sys.argv[3]

    # Call the inference function
    run_tensorrt_inference(engine_file_path, image_path, onnx_model_path)