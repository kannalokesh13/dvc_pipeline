import os
import cv2
from pathlib import Path

from yolov9 import YOLOv9


def get_detector(args):
    weights_path = args.weights
    classes_path = 'metadata.yaml'
    source_path = args.source
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

def inference_on_image(args):
    print("[INFO] Intialize Model")
    detector = get_detector(args)
    image = cv2.imread(args.source)

    print("[INFO] Inference Image")
    detections = detector.detect(image)
    detector.draw_detections(image, detections=detections)

    output_path = f"output/{Path(args.source).name}"
    print(f"[INFO] Saving result on {output_path}")
    cv2.imwrite(output_path, image)



if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Argument for YOLOv9 Inference using ONNXRuntime")

    parser.add_argument("--source", type=str, required=True, help="Path to image or video file")
    parser.add_argument("--weights", type=str, required=True, help="Path to yolov9 onnx file")
  
    args = parser.parse_args()

    #if args.image:
    inference_on_image(args=args)
   