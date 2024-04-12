import threading

import cv2


def rescale_coords(m_coords, scale):
    m_coords[0] *= 2
    m_coords[1] *= 2
    m_coords[2] = m_coords[0] + (m_coords[2] - m_coords[0]) * scale
    m_coords[3] = m_coords[1] + (m_coords[3] - m_coords[1]) * scale
    return m_coords


def process_frame(patch_frame, frame, m_coords, m_max_size, scale):
    x_centre = int((m_coords[2] - m_coords[0]) / 2 + m_coords[0])
    if m_coords[2] - m_coords[0] > m_max_size:
        m_coords[2] = x_centre + m_max_size / 2
        m_coords[0] = x_centre - m_max_size / 2
    half_width = int((m_coords[2] - m_coords[0]) / 2)

    y_centre = int((m_coords[3] - m_coords[1]) / 2 + m_coords[1])
    if m_coords[3] - m_coords[1] > m_max_size:
        m_coords[3] = y_centre + m_max_size / 2
        m_coords[1] = y_centre - m_max_size / 2
    half_height = int((m_coords[3] - m_coords[1]) / 2)

    patch = patch_frame[y_centre - half_height:y_centre + half_height, x_centre - half_width:x_centre + half_width]
    half_patch = cv2.resize(patch, None, fx=scale, fy=scale)
    half_frame = cv2.resize(frame, None, fx=scale, fy=scale)

    matches = cv2.matchTemplate(half_frame, half_patch, cv2.TM_SQDIFF_NORMED)
    _, _, location, _ = cv2.minMaxLoc(matches)

    min_location = [int(location[0] / scale), int(location[1] / scale)]

    return [min_location[0], min_location[1], min_location[0] + half_width * 2, min_location[1] + half_height * 2]


class AsyncBuffer:
    def __init__(self, max_size, matching_scale, step):
        self.max_size = max_size
        self.matching_scale = matching_scale
        self.step = step

        self.buffer = []
        self.size = 0
        self.i = 0

        self.coords = []

        self.thread = None
        self.lock = threading.Lock()

        self.finished = False

    def run(self, m_coords, m_max_size, m_matching_scale, step):
        self.i = 0
        while self.i < self.size:
            m_coords = process_frame(self.buffer[self.i - 1], self.buffer[self.i], m_coords, m_max_size, m_matching_scale)
            self.i = self.i + step
        with self.lock:
            self.coords = m_coords
            self.finished = True

    def run_async(self, m_coords):
        self.finished = False
        self.size = len(self.buffer)
        self.thread = threading.Thread(target=self.run, args=(m_coords, self.max_size, self.matching_scale, self.step,))
        self.thread.start()

    def is_finished(self):
        return self.finished

    def get_i(self, m_i):
        return self.buffer[m_i]

    def get(self):
        if self.is_finished():
            if self.thread is not None:
                self.thread.join()
            return self.coords
        else:
            return None

    def append(self, frame):
        if self.is_finished():
            self.buffer.append(frame)
        else:
            with self.lock:
                self.buffer.append(frame)
                self.size = self.size + 1

    def reset(self):
        self.buffer = []
        self.size = 0
        self.i = 0
