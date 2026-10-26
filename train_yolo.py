import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import os

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on road damage dataset')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    # Verify data.yaml exists
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f'Error: {data_path} not found. Run the annotation conversion first.')
        return

    # Load model
    model = YOLO('yolov8n.pt')  # pretrained nano model

    # Train
    model.train(data=str(data_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=0, seed=42, project='runs', name='road_damage_yolov8')

if __name__ == '__main__':
    main()
