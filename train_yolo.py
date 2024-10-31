from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="leaderboard.yaml",
                          epochs=50, imgsz=512, workers=6, name='yolo_clean')