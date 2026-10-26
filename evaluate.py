import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import os

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 model on validation set')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml')
    parser.add_argument('--weights', type=str, default='runs/detect/runs/road_damage_yolov8-2/weights/best.pt', help='Path to model checkpoint')
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f'Error: {data_path} not found.')
        return
    if not Path(args.weights).is_file():
        print(f'Error: weights {args.weights} not found.')
        return

    # Load model
    model = YOLO(args.weights)

    # Run validation
    metrics = model.val(data=str(data_path), batch=4, imgsz=640, device='cpu', save_json=True, project='runs', name='road_damage_yolov8')
    # The YOLO.val method prints a summary; we can also access metrics.box.map50 etc.
    print('--- Validation Summary ---')
    for k, v in metrics.results_dict.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    main()
