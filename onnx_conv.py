import os
import sys
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
import onnxruntime as ort
import numpy as np
import cv2
import time


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, classes = torch.max(x[:, :, 4:], 2, keepdim=True)
        classes = classes.float()
        return boxes, scores, classes


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def yolov8_export(file_path, device):
    model = YOLO(file_path)
    model = deepcopy(model.model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = True
            m.export = True
            m.format = 'onnx'
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return model


def export_onnx(file_path):
    size = [640]
    opset =16
    simplify = True
    dynamic = True
    batch =1

    print('\nStarting: %s' % file_path)

    print('Opening YOLOv8 model\n')

    device = select_device('cpu')
    model = yolov8_export(file_path, device)

    if len(model.names.keys()) > 0:
        print('\nCreating labels.txt file')
        f = open('labels.txt', 'w')
        for name in model.names.values():
            f.write(name + '\n')
        f.close()

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = size * 2 if len(size) == 1 else size

    onnx_input_im = torch.zeros(batch, 3, *img_size).to(device)
    onnx_output_file = os.path.basename(file_path).split('.pt')[0] + '.onnx'

    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'boxes': {
            0: 'batch'
        },
        'scores': {
            0: 'batch'
        },
        'classes': {
            0: 'batch'
        }
    }

    print('\nExporting the model to ONNX')
    torch.onnx.export(model, onnx_input_im, onnx_output_file, verbose=False, opset_version=opset,
                      do_constant_folding=True, input_names=['input'], output_names=['boxes', 'scores', 'classes'],
                      dynamic_axes=dynamic_axes if dynamic else None)

    if simplify:
        print('Simplifying the ONNX model')
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print('Done: %s\n' % onnx_output_file)
    return onnx_output_file



def preprocess(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, ...]
    return img.astype(np.float32)


def post_process(original_img, boxes, scores, classes):
    img = cv2.resize(original_img, (640, 640))
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2

    conf_thres = 0.25
    iou_thres = 0.45
    index = cv2.dnn.NMSBoxes(boxes.tolist(), scores.flatten().tolist(), conf_thres, iou_thres)

    for bbox, cls, score in zip(boxes[index], classes[index], scores[index]):
        label = f'{int(cls)}, {score[0]:.2f}'
        xywh = [int(i) for i in bbox]


        cv2.rectangle(img, (xywh[0], xywh[1], xywh[2], xywh[3]), (0, 255, 0), 2)
        cv2.putText(img, label, (xywh[0], xywh[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img

def onnx_inference_v8(onnx_path,img_path):
    img = cv2.imread(img_path)
    inputs = preprocess(img)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inputs = {sess.get_inputs()[0].name: inputs}
    outputs = sess.run(None, inputs)
    boxes, scores, classes = outputs

    boxes = boxes.squeeze(axis=0)
    scores = scores.squeeze(axis=0)
    classes = classes.squeeze(axis=0)
    output_img = post_process(img, boxes, scores, classes)
    output_path = 'outputonnx.jpg'
    cv2.imwrite(output_path, output_img)
    print(f"Image with bounding boxes saved to: {output_path}")
    return output_img

def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream YOLOv8 conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args


if __name__ == '__main__':
    export_onnx('yolov8n.pt')
    onnx_inference_v8('yolov8n.onnx','uploads/sample_image.jpeg')
    #args = parse_args()
    #sys.exit(main(args))