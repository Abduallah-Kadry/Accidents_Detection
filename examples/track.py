# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

from pathlib import Path
import torch
import argparse
import numpy as np
import cv2
from types import SimpleNamespace
import math
import os
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from boxmot.utils import logger as LOGGER
from boxmot.utils.torch_utils import select_device

tr = TestRequirements()
tr.check_packages(('ultralytics',))  # install

from ultralytics.yolo.engine.model import YOLO, TASK_MAP

from ultralytics.yolo.utils import SETTINGS, colorstr, ops, is_git_dir, IterableSimpleNamespace
from ultralytics.yolo.utils.checks import check_imgsz, print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.data.utils import VID_FORMATS

from multi_yolo_backend import MultiYolo
from utils import write_MOT_results

from collections import deque

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
line = [(100, 500), (1050, 500)]


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    predictor.args.tracking_config = \
        ROOT / \
        'boxmot' / \
        opt.tracking_method / \
        'configs' / \
        (opt.tracking_method + '.yaml')
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.args.tracking_method,
            predictor.args.tracking_config,
            predictor.args.reid_model,
            predictor.device,
            predictor.args.half
        )
        predictor.trackers.append(tracker)


@torch.no_grad()
def run(args):
    model = YOLO(args['yolo_model'] if 'v8' in str(args['yolo_model']) else 'yolov8n')
    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)

    # extract task predictor
    predictor = model.predictor

    # combine default predictor args with custom, preferring custom
    combined_args = {**predictor.args.__dict__, **args}
    # overwrite default args
    predictor.args = IterableSimpleNamespace(**combined_args)
    predictor.args.device = select_device(args['device'])
    LOGGER.info(args)

    # setup source and model
    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=False)
    predictor.setup_source(predictor.args.source)

    predictor.args.imgsz = check_imgsz(predictor.args.imgsz, stride=model.model.stride, min_dim=2)  # check image size
    predictor.save_dir = increment_path(Path(predictor.args.project) / predictor.args.name,
                                        exist_ok=predictor.args.exist_ok)

    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (predictor.save_dir / 'labels' if predictor.args.save_txt else predictor.save_dir).mkdir(parents=True,
                                                                                                 exist_ok=True)
    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(
            imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz))
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = 0, [], None, (
        ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
    predictor.add_callback('on_predict_start', on_predict_start)
    predictor.run_callbacks('on_predict_start')
    model = MultiYolo(
        model=model.predictor.model if 'v8' in str(args['yolo_model']) else args['yolo_model'],
        device=predictor.device,
        args=predictor.args
    )
    # Loop over frames
    trail_ids = {}
    extracted_path = str(predictor.save_dir) + '\\Extracted frames\\'
    for frame_idx, batch in enumerate(predictor.dataset):
        # for training purposes on batch of 100 frames
        # if frame_idx % 100 == 0:
        #     trail_ids = {}

        predictor.run_callbacks('on_predict_batch_start')
        predictor.batch = batch
        path, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, exist_ok=True,
                                   mkdir=True) if predictor.args.visualize and (
            not predictor.dataset.source_type.tensor) else False

        n = len(im0s)
        predictor.results = [None] * n

        # Preprocess
        with predictor.profilers[0]:
            im = predictor.preprocess(im0s)

        # Inference
        with predictor.profilers[1]:
            preds = model(im, im0s)

        # Postprocess moved to MultiYolo
        with predictor.profilers[2]:
            predictor.results = model.postprocess(path, preds, im, im0s, predictor)
        predictor.run_callbacks('on_predict_postprocess_end')

        # Visualize, save, write results
        # Loop over tracks in the current frame
        # for i in range(n):
        i = 0
        if predictor.dataset.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
            continue

        p, im0 = path[i], im0s[i].copy()
        p = Path(p)

        with predictor.profilers[3]:
            # get raw bboxes tensor
            dets = predictor.results[i].boxes.data
            # get tracker predictions
            predictor.tracker_outputs[i] = predictor.trackers[i].update(dets.cpu().detach().numpy(), im0)

        predictor.results[i].speed = {
            'preprocess': predictor.profilers[0].dt * 1E3 / n,
            'inference': predictor.profilers[1].dt * 1E3 / n,
            'postprocess': predictor.profilers[2].dt * 1E3 / n,
            'tracking': predictor.profilers[3].dt * 1E3 / n
        }

        # filter boxes masks and pose results by tracking results
        model.filter_results(i, predictor)
        # overwrite bbox results with tracker predictions
        model.overwrite_results(i, im0.shape[:2], predictor)

        # write inference results to a file or directory
        if predictor.args.verbose or predictor.args.save or predictor.args.save_txt or predictor.args.show:
            s += predictor.write_results(i, predictor.results, (p, im, im0))
            predictor.txt_path = Path(predictor.txt_path)

            # write MOT specific results
            if predictor.args.source.endswith(VID_FORMATS):
                predictor.MOT_txt_path = predictor.txt_path.parent / p.stem
            else:
                # append folder name containing current img
                predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.name

            if predictor.tracker_outputs[i].size != 0 and predictor.args.save_txt:
                write_MOT_results(predictor.MOT_txt_path, predictor.results[i], frame_idx, i, )

        # TODO input the im0 to the vit after transformation (we will see the predict function of the Vit) it should output the Feature flattened vector
        # Draw the tracks in black screen
        black_frame = np.zeros((224, 224, 3))

        if len(predictor.tracker_outputs[i]) > 0:
            bbox_xyxy = predictor.tracker_outputs[i][:, :4]
            identities = predictor.tracker_outputs[i][:, -1]
            object_id = predictor.tracker_outputs[i][:, 4]

            black_frame = draw_trails(black_frame, bbox_xyxy, object_id, trail_ids, frame_idx, identities)

            print(black_frame.shape)
            print(type(black_frame))
            print(type(torch.tensor(black_frame)))
            print(torch.tensor(black_frame).shape)

            # to save the trails of the video (black_frames
            save_path = extracted_path + f'frame_{frame_idx}.jpg'

            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path)
            # cv2.imshow('black_frame', black_frame)
            cv2.imwrite(save_path, black_frame)
            # cv2.imshow('frame_with_trail', draw_trails(black_frame, bbox_xyxy, object_id, trail_ids, frame_idx, identities))
            cv2.waitKey(1)  # 1 millisecond

            # display an image in a window using OpenCV imshow()
            # responsible for displaying the current frame
            if predictor.args.show and predictor.plotted_img is not None:
                predictor.show(p.parent)

            # save video predictions
            if predictor.args.save and predictor.plotted_img is not None:
                predictor.save_preds(vid_cap, i,
                                     str(predictor.save_dir / p.name))  # save the frames to this specific path

            predictor.run_callbacks('on_predict_batch_end')

            # print time (inference-only)
            if predictor.args.verbose:
                LOGGER.info(
                    f'{s}YOLO {predictor.profilers[1].dt * 1E3:.1f}ms, TRACKING {predictor.profilers[3].dt * 1E3:.1f}ms')

        # TODO input the black trails into the VGG 16 and save the output feature vector after flattening i think

        # TODO we should make the combined feature vectors that should 


    # Release assets
    if isinstance(predictor.vid_writer[-1], cv2.VideoWriter):
        predictor.vid_writer[-1].release()  # release final video writer

    # Print results
    if predictor.args.verbose and predictor.seen:
        t = tuple(x.t / predictor.seen * 1E3 for x in predictor.profilers)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess, %.1fms tracking per image at shape '
        f'{(1, 3, *predictor.args.imgsz)}' % t)
    if predictor.args.save or predictor.args.save_txt or predictor.args.save_crop:
        nl = len(list(predictor.save_dir.glob('labels/*.txt')))  # number of labels
    s = f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}" if predictor.args.save_txt else ''
    LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

    predictor.run_callbacks('on_predict_end')


def draw_trails(img, bbox, object_id, trail_ids, current_frame, identities=None, offset=(0, 0)):
    # remove tracked point from buffer if object is lost
    # This line should be modified to pop out from the identities queue after at least 3 frames not immediately
    #
    for key, trail in list(trail_ids.items()):
        if key not in object_id:
            if len(trail) > 1:
                trail.pop()
                trail.pop()
            else:
                trail.pop()
            if len(trail) == 0:
                trail_ids.pop(key)

    for i, box in enumerate(bbox):
        # get bbox dimensions
        x1, y1, x2, y2 = [int(i) for i in box]
        # modify for dynamic scaling
        x1 = x1 * (224 / 1280)
        x2 = x2 * (224 / 1280)
        y1 = y1 * (224 / 720)
        y2 = y2 * (224 / 720)

        point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        id = int(object_id[i]) if object_id is not None else 0

        if id not in trail_ids:
            trail_ids[id] = deque(maxlen=300)  # trail of that box

        trail_ids[id].appendleft(point)

    for key, trail in trail_ids.items():  # Iterate over trail_ids instead of object_id
        if len(trail) == 0:
            continue
        # Draw trail
        for j in range(1, len(trail)):
            if trail[j] is None:
                continue
            # thickness = int(np.sqrt(64 / float(j * 1.5)) * 1.5)
            # Set the head color to red (BGR: 0, 0, 255)
            # cv2.line(img, trail[j - 1], trail[j], color, thickness)
            # c = max(int(250 + 30 - j * 30), 40)
            # cv2.line(img, trail[j - 1], trail[j], (0, 0, c), 3)

            cv2.circle(img, trail[j], radius=3, color=(255, 255, 255), thickness=-1)
            if j == len(trail) - 1 and key in object_id:
                cv2.circle(img, trail[1], radius=3, color=(0, 0, 255), thickness=-1)

    return img


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')

    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', help='display tracking video results')
    parser.add_argument('--save', action='store_true', help='save video tracking results')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--hide-label', action='store_true', help='hide labels when show')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true', help='save tracking results in a txt file')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
