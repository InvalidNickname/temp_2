import time

import cv2
import numpy as np
import torch

from video_provider import VideoProvider
from mobile_net import MobileNet
from async_buffer import AsyncBuffer
from async_buffer import process_frame

fps = 24
video_path = "./abrams/video.mp4"
weights_path = "./abrams/model.model"
device = "cpu"
max_size = 200
threshold = 0.9
matching_scale = 0.2
buffer_step = 2

draw_output = False
debug = True
save_to_file = False
save_path = "./processed/"

if __name__ == '__main__':
    video = VideoProvider(video_path)

    model = MobileNet(weights_path, device, threshold)

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

    if debug:
        fps_history.pop(0)
        print("Средний fps: " + str(np.mean(fps_history)) + ", минимальный: " + str(np.min(fps_history)))
