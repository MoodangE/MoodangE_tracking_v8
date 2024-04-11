import argparse
import cv2
import numpy as np
import os
import platform
import shutil
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor
from pathlib import Path
from collections import defaultdict

from ultralytics.data.augment import LetterBox

# SORT
from sort.sort import Sort

# YOLOv8
from ultralytics import YOLO
from ultralytics.data.loaders import LoadStreams, LoadImagesAndVideos
from ultralytics.utils import ops, LOGGER, colorstr
from ultralytics.utils.checks import check_requirements, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, time_sync

# MoodangE
from moodangE.congestion import calculate_congestion

# Parameters
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
TEXT_BLACK = (0, 0, 0)
TEXT_WHITE = (255, 255, 255)
PALETTE = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


# Bounding box color selector from id
def compute_color_for_labels(box_id):
    color = [int((p * (box_id ** 2 - box_id + 1)) % 255) for p in PALETTE]
    return tuple(color)


# Draw Bounding box and Label
def draw_boxes(img, boxes, identities, categories, names, blur, duration_data):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(i) for i in box]

        # Bounding box id, category and color
        box_id = int(identities[i]) if identities is not None else 0
        box_category = categories[i] if categories is not None else 0
        box_color = compute_color_for_labels(box_id)
        duration_data.append(box_id)

        # Bounding box Blur
        if blur:
            target_frame = img[y1:y2, x1:x2]
            target_frame = cv2.GaussianBlur(target_frame, (64, 64), 9)
            img[y1:y2, x1:x2] = target_frame

        # Bounding box Labeling
        label = f'{names[box_category]} {box_id}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + label_size[0], y1 - label_size[1] - 5), box_color, -1)
        cv2.putText(img, label, (x1, y1 - label_size[1] + 8), cv2.FONT_HERSHEY_PLAIN, 1, TEXT_WHITE, 1)

    return img, duration_data


# Bounding box footer Tracking
def draw_tracking(img, boxes, identities, tracking_path, tracking_id):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(i) for i in box]

        # Bounding box id, category and color
        box_id = int(identities[i]) if identities is not None else 0
        box_color = compute_color_for_labels(box_id)
        cv2.circle(img, (int((x1 + x2) / 2), y2), 2, box_color, -1)

        # Record tracking Path
        tracking_path[box.id].append((int((x1 + x2) / 2), y2))

        # Record tracking Person id
        if id not in tracking_id:
            tracking_id.append(box_id)
            start_point = (int((x1 + x2) / 2), y2)
            end_point = (int((x1 + x2) / 2), y2)
            cv2.line(img, start_point, end_point, box_color, 2)
        else:
            line = len(tracking_path[box_id])
            for point in range(line - 1):
                start_point = tracking_path[box_id][point]
                end_point = tracking_path[box_id][point + 1]
                cv2.line(img, start_point, end_point, box_color, 2)

    return img


def draw_text(img, congestion, person):
    input_text = f"Level: {congestion}, Waiting: {person}"
    cv2.putText(img, input_text, (10, 50), cv2.FONT_ITALIC, 2, TEXT_WHITE, 16, cv2.LINE_AA)
    cv2.putText(img, input_text, (10, 50), cv2.FONT_ITALIC, 2, TEXT_BLACK, 4, cv2.LINE_AA)
    return img


@torch.no_grad()
def run(
        # YOLOv8 params
        weight="models/yolov8n-person.pt",
        source='ultralytics/assets/bus.jpg',  # file/dir/URL/glob, 0 for webcam
        no_save=False,  # do not save image or video
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        project='inference_person',  # save results to project
        name='exp',  # save results to name
        conf_thresh=0.25,  # confidence threshold
        iou_thresh=0.45,  # NMS IOU threshold
        agnostic=False,  # class-agnostic NMS=False,  # class-agnostic NMS
        view_img=False,  # show results

        # SORT params
        sort_max_age=30,
        sort_min_hits=2,
        sort_iou_thresh=0.2,

        # MoodangE params
        blur=False,  # bounding box blur
        tracking=False,  # tracking path visualize
        duration=5.0,  # update cycle duration
):
    source = str(source)
    save_img = not no_save and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # check file type
    webcam = source.isnumeric() or source.endswith('.txt')

    # Initialize SORT
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)

    # Directory and CUDA settings
    device = select_device(device)
    save_dir = increment_path(Path(project) / name)  # increment run
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)  # delete output folder
    os.makedirs(save_dir)  # make new output folder

    # Load model
    model = YOLO(weight)
    model.overrides['conf'] = conf_thresh
    model.overrides['iou'] = iou_thresh
    model.overrides['agnostic_nms'] = agnostic
    names = model.names

    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImagesAndVideos(source)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Congestion setting
    congestion_level = 'None'
    waiting_person = 0
    duration_data = []
    duration_time = 0.0
    duration_frame = 0
    tracking_id = []
    tracking_path = defaultdict(list)

    # Progress predictions
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for frame_idx, (path, im, s) in enumerate(dataset):
        t1 = time_sync()

        # Inference
        pred = model(im)

        # Process prediction
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, s, im0 = Path(path[i]), '%g: ' % i, im[i].copy()
            save_path = str(save_dir / p.name)  # img.jpg

            for c in det.boxes.cls.unique():
                n = (det.boxes.cls == c).sum()
                s += f" - {n} {names[int(c)]}"

            # Pass detections to SORT
            det_to_sort = np.empty((0, 6))
            for x1, y1, x2, y2, conf, cls in det.boxes.data.cpu().numpy():
                det_to_sort = np.vstack((det_to_sort, np.array([x1, y1, x2, y2, conf, cls])))

            # Run SORT
            track_det = sort_tracker.update(det_to_sort)

            # Draw bounding boxes and tracking path
            if len(track_det) > 0:
                bbox_xyxy = track_det[:, :4]
                identities = track_det[:, 8]
                categories = track_det[:, 4]
                im0, duration_data = draw_boxes(im0, bbox_xyxy, identities, categories, names, blur, duration_data)
                im0 = draw_tracking(im0, bbox_xyxy, identities, tracking_path, tracking_id) if tracking else im0

            # Draw Congestion Level
            draw_text(im0, congestion_level, waiting_person)

            # Stream results
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if vid_writer[i] is not None and isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if dataset.mode:  # video
                            fps = dataset.cap.get(cv2.CAP_PROP_FPS)
                            w = int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                    if vid_writer[i] is not None:
                        vid_writer[i].write(im0)

            # Progress, Time taken per frame
            total_duration = time_sync() - t1
            print('{} | Time taken per frame: {:.2f}s'.format(s, total_duration))

        #  Update Congestion Level
        duration_time += time_sync() - t1
        duration_frame += 1
        if duration_time >= duration:
            congestion_level, waiting_person = calculate_congestion(duration_data, duration_frame)
            print('Congestion Level: {},\t Waiting Person: {}'.format(congestion_level, waiting_person))

            # Reset
            duration_data, duration_time, duration_frame = [], 0.0, 0

    # Print Result
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image' % t)
    if save_img:
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()

    # YOLOv8 params
    parser.add_argument('--weight', type=str, default='models/yolov8n-person.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='ultralytics/assets/bus.jpg',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--no-save', action='store_true', help='do not save images/videos')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='inference_person', help='save results to project')
    parser.add_argument('--name', default='exp', help='save results to name')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thresh', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--agnostic', action='store_true', help='augmented inference')
    parser.add_argument('--view-img', action='store_true', help='show results')

    # SORT params
    parser.add_argument('--sort-max-age', type=int, default=30,
                        help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--sort-min-hits', type=int, default=2,
                        help='start tracking only after n number of objects detected')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.2,
                        help='intersection-over-union threshold between two frames for association')

    # Person params
    parser.add_argument('--blur', action='store_true', help='bounding box blur')
    parser.add_argument('--tracking', action='store_true', help='tracking path visualize')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Designated as 4 seconds based on the image of FPS 30.')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    check_requirements()
    opt = parse_opt()
    run(**vars(opt))
