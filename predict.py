import argparse
import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np

def draw_boxes(img, results, class_names, conf_thres):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if conf < conf_thres:
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            label = f"{class_names[cls_id]} {conf:.2f}"
            # draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # put label
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def save_yolo_txt(txt_path, results, conf_thres):
    lines = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf)
            if conf < conf_thres:
                continue
            cls_id = int(box.cls)
            xywh = box.xywh[0].cpu().numpy()  # (cx, cy, w, h) normalized by img size
            line = f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}"
            lines.append(line)
    txt_path.write_text('\n'.join(lines))

def main():
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on road damage images')
    parser.add_argument('--source', type=str, default='data/raw_dataset/images', help='Path to image or directory of images (default: raw images for demo)')
    parser.add_argument('--weights', type=str, default='runs/detect/runs/road_damage_yolov8-2/weights/best.pt', help='Path to trained model weights')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for displaying boxes')
    parser.add_argument('--output-dir', type=str, default='sample_results/predictions', help='Directory to save annotated images and txt files')
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f'Error: source path {source_path} does not exist')
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    class_names = model.names  # dictionary id -> name

    # collect image files
    if source_path.is_dir():
        img_paths = list(source_path.rglob('*.jpg')) + list(source_path.rglob('*.png')) + list(source_path.rglob('*.jpeg'))
    else:
        img_paths = [source_path]

    if not img_paths:
        print('No image files found in the specified source')
        return

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'Warning: Failed to read image {img_path}')
            continue
        results = model.predict(str(img_path), conf=args.conf_thres, save=False)
        annotated = draw_boxes(img.copy(), results, class_names, args.conf_thres)
        # save annotated image
        out_img_path = output_dir / (img_path.stem + '_pred.jpg')
        cv2.imwrite(str(out_img_path), annotated)
        # save YOLO txt label
        out_txt_path = output_dir / (img_path.stem + '.txt')
        save_yolo_txt(out_txt_path, results, args.conf_thres)
        print(f'Processed {img_path.name} -> {out_img_path.name}')

if __name__ == '__main__':
    main()
