import cv2


class VideoProvider:
    def __init__(self, file_path=None):
        if file_path is None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print("Невозможно открыть видеопоток")
            exit(0)

    def get(self):
        status, frame = self.cap.read()
        if status is None:
            return None
        else:
            return frame
