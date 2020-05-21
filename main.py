import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans

video_name = "mov.mp4"
output_name = "out-kmeans.jpg"
output_height = 680 # in px
output_width = 1920 # Squeezing happens in this dimension
use_avg_instead = False # If true, compute the average color of each frame instead of the dominant. 
squash = True # If true, compress all frame slices into OUTPUT_WIDTH total slices.
skipped_frames = 24 # Number of frames to skip between slices.

frames = [] 
output_img = []

def dominant_color(frame):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(frame)
    print(kmeans.cluster_centers_)
    print(kmeans.labels_, len(kmeans.labels_))
    # return kmeans.cluster_centers_[0]

# Squeeze frames to OUTPUT_WIDTH total frames
def squash():
    if output_width <= len(frames):
        return 
    ratio = len(frames) // output_width

def main():
    img = cv2.imread(output_name, )
    video = cv2.VideoCapture(video_name)
    if not video.isOpened():
        print("Error opening video")
        sys.exit()

    ret, frame = video.read()
    while ret:
        # Downsize by 8x, flatten 2D pixel array to 1D list of pixel values
        frame = cv2.resize(frame, (frame.shape[0]//8, frame.shape[1]//8))
        frame = np.reshape(frame, (frame.size//3, 3))
        if use_avg_instead:
            frames.append(np.mean(frame, axis=0))
        else:
            frames.append(dominant_color(frame))
        # break
        for i in range(0, skipped_frames):
            ret, frame = video.read()
            if not ret:
                break
    video.release()

    for f in frames:
        output_img.append([f]*output_width)
    # print(output_img)
    # print(type(output_img[0]))
    # print(output_img[0])
    cv2.imwrite("out.png", np.asarray(output_img))

if __name__ == "__main__":
    main()