import cv2

def video_read(path):
    reader = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def video_write(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(path, fourcc, 20.0, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(frame)
    writer.release()

