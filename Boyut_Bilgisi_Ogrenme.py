import cv2
import os
import glob

folder_path = 'C:/Users/Mert/Desktop/VISEA/Train/Images'
jpg_files = glob.glob(os.path.join(folder_path, '*.png'))

max_width = 0
max_height = 0
max_file = None
min_width = float("inf")
min_height = float("inf")
min_file = None

with open('image_orjinal_resim_boyutları.txt', 'w') as f:
    for index, jpg_file in enumerate(jpg_files):
        img = cv2.imread(jpg_file)
        height, width = img.shape[:2]
        f.write(f"{index + 1}. Dosya: {jpg_file}, Genişlik: {width} piksel, Yükseklik: {height} piksel\n")
        if width > max_width and height > max_height:
            max_width = width
            max_height = height
            max_file = jpg_file
        if width < min_width and height < min_height:
            min_width = width
            min_height = height
            min_file = jpg_file

    f.write(f"En büyük resim: {max_file}, Genişlik: {max_width} piksel, Yükseklik: {max_height} piksel\n")
    f.write(f"En küçük resim: {min_file}, Genişlik: {min_width} piksel, Yükseklik: {min_height} piksel")
