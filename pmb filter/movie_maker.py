import cv2
import os

from config import outputs_dir

video_filename = '0019'

image_folder = os.path.join(outputs_dir,video_filename)
video_name =  image_folder + '/' + video_filename + '.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

images.sort(key=lambda f: int(os.path.splitext(f)[0]))

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 3, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

print('Done!')