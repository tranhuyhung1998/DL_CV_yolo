import os
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2


def build_bboxes(bbox_path='data/Public_leaderboard_data/test1_bbox.txt'):

    bbox_dict = {}
    with open(bbox_path) as f:
        for line in f:
            idx, box = line.split(':')
            ct, slice, organ = [part.strip() for part in idx.strip()[1:-1].split(',')]
            ct = f"{ct:02}"
            slice += '.png'
            organ = int(organ)
            if (ct, slice) not in bbox_dict:
                bbox_dict[(ct, slice)] = {}

            x0, y0, x1, y1 = [int(part.strip()) for part in box.strip()[1:-1].split(',')]
            bbox_dict[(ct, slice)][int(organ)] = np.array([x0, y0, x1, y1])
    return bbox_dict


if __name__ == '__main__':

    # Load a model
    model = YOLO("runs/segment/yolo_clean/weights/best.pt")
    test_path = 'data/Public_leaderboard_data/test1_images_clean/'
    pred_path = 'data/Public_leaderboard_data/test_labels/'

    bbox_dict = build_bboxes()

    for folder in os.listdir(test_path):
        print(f'Processing {test_path}/{folder}')
        os.makedirs(os.path.join(pred_path, folder), exist_ok=True)
        imgs = os.listdir(os.path.join(test_path, folder))
        results = model([os.path.join(test_path, folder, img)
                         for img in imgs])
        for img, res in zip(imgs, results):
            pred = np.zeros((512, 512), dtype=np.uint8)
            if res.masks is not None:
                labels = res.boxes.cls.cpu().numpy().astype(int) + 1
                masks = res.masks.data.cpu().numpy()

                bboxes = bbox_dict[(folder, img)]

                for idx, obj_id in enumerate(labels):
                    if obj_id not in bboxes:
                        continue
                    x1, y1, x2, y2 = bboxes[obj_id]

                    mask = masks[idx]

                    mask[:y1, :] = 0
                    mask[y2 + 1:, :] = 0
                    mask[:, :x1] = 0
                    mask[:, x2 + 1:] = 0

                    pred[mask > 0] = obj_id

            image = Image.fromarray(pred)
            image.save(os.path.join(pred_path, folder, img))

    # # Validate the model
    # metrics = model.val()  # no arguments needed, dataset and settings remembered
    # print(metrics)

