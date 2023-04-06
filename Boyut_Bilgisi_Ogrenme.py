"""
Bu Python kodu, belirtilen bir klasördeki tüm PNG dosyalarının boyutlarını kontrol eder ve en büyük ve en küçük boyutlu dosyaları belirler. 
Ayrıca, her bir dosyanın orijinal boyutunu bir metin dosyasına yazar.

İlk olarak, "cv2" ve "os" kütüphaneleri ve "glob" modülü içe aktarılır. Ardından, klasör yolunu belirten bir değişken tanımlanır ve "glob.glob" kullanılarak 
klasördeki tüm PNG dosyaları bir listeye aktarılır.

Sonra, "max_width", "max_height", "max_file" değişkenleri en büyük boyutlu dosyayı ve "min_width", "min_height", "min_file" değişkenleri en küçük boyutlu 
dosyayı belirlemek için kullanılır. Ardından, her bir PNG dosyasının boyutunu kontrol etmek için bir döngü oluşturulur. Her dosyanın orijinal boyutu bir 
metin dosyasına yazılır ve en büyük ve en küçük boyutlu dosyalar belirlenir.

Son olarak, en büyük ve en küçük boyutlu dosyaların yolu ve boyutu da metin dosyasına yazılır.

Bu kod, görüntü işleme uygulamaları için kullanılabilen bir temel bilgi sağlar ve veri setindeki görüntülerin boyutları hakkında fikir edinmek için kullanılabilir.

"""

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
