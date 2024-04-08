import cv2
from sort.sort import Sort

from ultralytics import YOLO
from ultralytics.utils import ops

#create instance of SORT
mot_tracker = Sort()

# get detections
model_path = "models/yolov8n-person.pt"
yolov8 = YOLO(model_path)

# update SORT
track_bbs_ids = mot_tracker.update(detections)

# track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
