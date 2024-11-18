import argparse

from ultralytics import YOLO


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--dataset', type=str, help='dataset yaml file')
    # parser.add_argument('-m', '--model', type=str, help='model to run')
    # parser.add_argument('-e', '--epoch', type=int, help='epochs')
    # args = parser.parse_args()

    # Load a model
    # model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
    # model = YOLO("yolo11l-seg.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
    model = YOLO("runs/segment/yolo_clean_16/weights/best.pt")

    # Train the model
    results = model.train(data="leaderboard.yaml", name='yolo_clean_l_final_fantasy',
                          epochs=150, imgsz=512, workers=16, batch=32, 
                          cache=True, patience=30, 
                          fliplr=0.0, crop_fraction=0.2, degrees=15, translate=0.2)

    # model = YOLO(f'yolo11{args.model}-seg.pt')
    # model.train(data=f'{args.dataset}.yaml', name=f'yolo_{args.dataset}_{args.model}_max',
    #             epochs=args.epoch, imgsz=512, workers=16, batch=32, cache=True)
