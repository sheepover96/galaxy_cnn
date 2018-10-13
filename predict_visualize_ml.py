import sys
import csv
from astropy.io import fits
import aplpy
from PIL import Image
import numpy as np
from astropy.visualization import (ZScaleInterval,ImageNormalize)
import matplotlib.pyplot as plt

FILE_HOME = "/Users/sheep/Documents/research/project/hsc"

img_width = 100
img_height = 100

img_channels = 4
#raw_size = (48, 48, img_channels)
raw_size = (239, 239, img_channels)
#input_shape = (24, 24, img_channels)
input_shape = (50, 50, img_channels)

argv = sys.argv

inputfile = argv[1]
outputfile = argv[2]
f_read = open(inputfile, "r")
f_write = open(outputfile, "w")
reader = csv.reader(f_read)

f_write.write("<html>\n")
f_write.write("\t<table border='1'>\n")

#header = ['catalog id', 'g band', 'r band', 'i band', 'combined img', 'correct label', 'False probability', 'True Probability', 'answer']
header = ['catalog id', 'raw_image band2', 'band 1', 'band 2', 'band 3', 'band 4', 'correct label', 'answer']

ths = ''.join(['<th>%s</th>' % th for th in header])
f_write.write('\t\t\t%s\n' % ths)

np.set_printoptions(threshold=np.inf)

def to_list(string):
    return string[2:-2].split("', '")

def make_img_td(filepath):
    return '<td><img src="%s" width="%s" height="%s" title="%s"></td>' % (filepath, img_width, img_height, filepath)

#def normalize(image):
#    return (image - image.min()).astype(float)*255 / (image.max() - image.min()).astype(float)

def special_median_filter(src, ksize):
    # 畳み込み演算をしない領域の幅
    d = int((ksize - 1) / 2)
    h, w, c = src.shape[0], src.shape[1], src.shape[2]

    # 出力画像用の配列（要素は入力画像と同じ）
    dst = src.copy()
    result = src.copy()

    for i in range(c):
        for y in range(d, h - d):
            for x in range(d, w - d):
                # 近傍にある画素値の中央値を出力画像の画素値に設定
                dst[y][x][i] = np.median(src[y - d:y + d + 1, x - d:x + d + 1, i])

    means = []
    for i in range(c):
        means.append(np.mean(src[:, :, i] - dst[:, :, i]))

    for i in range(c):
        for y in range(d, h - d):
            for x in range(d, w - d):
                # 近傍にある画素値の中央値を出力画像の画素値に設定
                pixel = src[y, x, i]
                # print(pixel - dst[y, x, i])
                if pixel == 0 or pixel == 255:
                    result[y, x, i] = dst[y, x, i]
                elif pixel - dst[y, x, i] > means[i]:
                    result[y, x, i] = dst[y, x, i]
    return result


def median_filter(image, ksize):
    # 畳み込み演算をしない領域の幅
    d = int((ksize - 1) / 2)
    h, w = image.shape[0], image.shape[1]

    # 出力画像用の配列（要素は入力画像と同じ）
    dst = image.copy()

    for y in range(d, h - d):
        for x in range(d, w - d):
            # 近傍にある画素値の中央値を出力画像の画素値に設定
            dst[y][x] = np.median(image[y - d:y + d + 1, x - d:x + d + 1])

    return dst


def normalize(image):
    min_value = image.min()
    if min_value < 0:
        image = image - min_value
        min_value = 0
    image_center = zoom_img(image, image.shape[0], 5)
    max_value = image_center.max()
    #max_value = image.max()
    #normalized = (image - min_value + max_value/20.0).astype(float)*255 / (max_value - min_value + max_value/20.0).astype(float)
    normalized = (image - min_value).astype(float)*255 / (max_value - min_value).astype(float)
    normalized = np.clip(normalized, normalized.min(), 255)
    print("min = %s, max = %s" % (normalized.min(), normalized.max()))
    return normalized

def save_as_image(image, output_path):
    #image = normalize(image)
    #image = image + 1.0
    #image = np.where(image > 4, 4, image)
    #image = image * 255/4
    #image = np.where(image > 255, 255, image)
    #image = median_filter(image, 8)
    print(output_path)
    norm = ImageNormalize(image,interval=ZScaleInterval())
    #print(type(norm))
    plt.imshow(image, cmap='gray', interpolation='nearest', norm=norm)
    plt.savefig(output_path, dpi = 300, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    #pil_img = Image.fromarray(np.uint8(image + 1.0))
    #pil_img.save(output_path)

def save_as_images(datas):
    for idx, (image, output_path) in enumerate(datas):
        if idx == 0:
            concated_image = image
        else:
            concated_image = np.concatenate([concated_image, image])

    flat_image = concated_image.flatten()
    mean = flat_image.mean()
    std = flat_image.std()

    for idx, (image, output_path) in enumerate(datas):
        image = np.where(image < mean - 3*std, mean - 3*std, image)
        image = np.where(image > mean + 3*std, mean + 3*std, image)
        image = (image + 3*std - mean)*255/(6*std)
        #image = np.where(image < 0, 0, image)
        #image = np.where(image > 3.5, 0, image)
        pil_img = Image.fromarray(np.uint8(image))
        pil_img.save(output_path)

def zoom_img(img, original_size, pickup_size):
    startpos = int(original_size / 2) - int(pickup_size / 2)
    img = img[startpos:startpos+pickup_size, startpos:startpos+pickup_size]
    return img

def load_and_resize(filepath):
    try:
        raw_image = fits.getdata(filepath)
        image = raw_image
        #image = zoom_img(image, raw_size[0], input_shape[0])
        trimmed_image = zoom_img(image, raw_size[0], input_shape[0])
        return (image, trimmed_image)
    except FileNotFoundError:
        print(filepath)

def to_png_and_save(fits_paths):
    output_paths = []
    count = 0
    raw_image_path = None
    images = []
    for filepath in fits_paths:
        (image, trimmed_image) = load_and_resize(filepath)
        output_filepath = filepath.split('.')[0] + ".png"
        save_as_image(trimmed_image, output_filepath)
        images.append(( trimmed_image, output_filepath ))
        output_paths.append(output_filepath)
        if count == 1:
            raw_image_path = filepath.split('.')[0] + "raw.png"
            save_as_image(image, raw_image_path)
        #count += 1 
    #save_as_images(images)
    return (output_paths, raw_image_path)

for i, row in enumerate(reader):
   cat_id = row[0]
   paths = to_list(row[1])
   img_paths = []
   for path in paths:
       #replaced = path.replace('/disk/cos/ono', '/Users/daiz/disk/cos/ono')
       path = FILE_HOME + path
       img_paths.append(path)
   (png_img_paths, raw_image_path) = to_png_and_save(img_paths)
   img_tds = ''.join([make_img_td(filepath) for filepath in png_img_paths])
   label = row[3]
   #probabilities = [row[4], row[5]]
   answer = row[4]

   if int(answer) == int(label):
       color = "#2EFE64"
   else:
       color = "#F78181"

   f_write.write('\t\t<tr bgcolor="%s">\n' % color)
   f_write.write("\t\t\t<td>%s</td>\n" % cat_id)
   f_write.write('\t\t\t%s\n' % make_img_td(raw_image_path))
   f_write.write('\t\t\t%s\n' % img_tds)
   f_write.write("\t\t\t<td>%s</td>\n" % label)
   f_write.write('\t\t\t<td>%s</td>\n' % answer)
   f_write.write("\t\t</tr>\n")

   #print("No. %s finished" % i)

f_write.write("\t</table>\n")
f_write.write("</html>\n")
