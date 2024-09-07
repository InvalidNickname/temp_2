import argparse
import time

import cv2
import numpy as np
import torch

from video_provider import VideoProvider
from mobile_net import MobileNet
from async_buffer import AsyncBuffer
from async_buffer import process_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="Neural network type", choices=["ssd", "fastrcnn"])
    parser.add_argument("weights", type=str, help="Relative weights path")
    parser.add_argument("-p", type=str, help="Relative video path, only used when camera isn't a video provider")
    parser.add_argument("-s", type=int, help="Max window size for template matching, default = 200", default=200)
    parser.add_argument("-t", type=float, help="Neural network threshold, default = 0.4", default=0.4)
    parser.add_argument("-m", type=float, help="Matching scale, default = 0.2", default=0.2)
    parser.add_argument("-b", type=int, help="Buffer step size, default = 2", default=2)
    parser.add_argument("--cuda", help="Compute on CUDA", action=argparse.BooleanOptionalAction, dest="cuda", default=False)
    parser.add_argument("--debug", help="Enable debug output", action=argparse.BooleanOptionalAction, dest="d", default=False)
    parser.add_argument("--draw-debug", help="Draw debug output", action=argparse.BooleanOptionalAction, dest="dd", default=False)
    parser.add_argument("--save", help="Save processed images to file", action=argparse.BooleanOptionalAction, dest="save", default=False)
    parser.add_argument("--save-path", type=str, help="Relative save path for processed images", dest="save_path", default="./processed/")

    args = parser.parse_args()

    weights_path = args.weights
    video_path = args.p
    net = args.net
    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"
    debug = args.d
    draw_output = args.dd
    save_to_file = args.save
    save_path = args.save_path
    max_size = args.s
    threshold = args.t
    matching_scale = args.m
    buffer_step = args.b

    video = VideoProvider(video_path)
    fps = video.get_fps()

    model = MobileNet(weights_path, device, threshold, net)

    buffer = AsyncBuffer(max_size, matching_scale, buffer_step)
    coords = []

    test_frame = None
    fps_history = []

    frame_counter = 0

    while True:

        frame = video.get()
        if frame is None:
            print("Кадр не получен")
            break

        start = time.time()

        if draw_output or save_to_file:
            test_frame = frame.copy()
        frame_tensor = (torch.tensor(frame, device=device).float() / 255.0).permute(2, 0, 1).unsqueeze(0)

        if model.is_finished():
            # получены результаты нейросети
            result = model.get()
            if result is not None:
                # на кадре обнаружен объект
                coords = result
                # проход по части накопившихся кадров в буфере, определение смещения
                buffer.run_async(coords)
            model.run_async(frame_tensor)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if buffer.is_finished():
            # получено смещение за прошлые кадры
            if buffer.size > 0:
                coords = buffer.get()
            else:
                coords = process_frame(buffer.get_i(0), frame, coords, max_size, matching_scale)
            buffer.reset()

        buffer.append(frame)

        frame_time = time.time() - start

        if debug:
            cur_fps = 1 / frame_time
            fps_history.append(cur_fps)
            print("Время кадра: " + str(frame_time) + " сек, " + str(cur_fps) + " fps")

        if draw_output or save_to_file:
            if len(coords) > 0:
                cv2.rectangle(test_frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 3)
            if draw_output:
                cv2.imshow("Tracking", test_frame)
                cv2.waitKey(1)
            if save_to_file:
                cv2.imwrite(save_path + str(frame_counter) + ".png", test_frame)
                frame_counter = frame_counter + 1

        wait_time = 1 / fps - (time.time() - start)
        if wait_time > 0:
            time.sleep(wait_time)

    video.release()

    if debug:
        fps_history.pop(0)
        print("Средний fps: " + str(np.mean(fps_history)) + ", минимальный: " + str(np.min(fps_history)))
