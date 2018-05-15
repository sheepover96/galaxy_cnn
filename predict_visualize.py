import sys
import csv
from astropy.io import fits
import aplpy
from PIL import Image
import numpy as np

FILE_HOME = "/Users/sheep/Documents/research/project/hsc"

img_width = 100
img_height = 100

img_channels = 5
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
header = ['catalog id', 'raw_image band2', 'band 1', 'band 2', 'band 3', 'band 4', 'combined img', 'correct label', 'False probability', 'True Probability', 'answer']

ths = ''.join(['<th>%s</th>' % th for th in header])
f_write.write('\t\t\t%s\n' % ths)

def to_list(string):
    return string[2:-2].split("', '")

def make_img_td(filepath):
    return '<td><img src="%s" width="%s" height="%s" title="%s"></td>' % (filepath, img_width, img_height, filepath)

#def normalize(image):
#    return (image - image.min()).astype(float)*255 / (image.max() - image.min()).astype(float)

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
    image = normalize(image)
    pil_img = Image.fromarray(np.uint8(image))
    pil_img.save(output_path)

def zoom_img(img, original_size, pickup_size):
    startpos = int(original_size / 2) - int(pickup_size / 2)
    img = img[startpos:startpos+pickup_size, startpos:startpos+pickup_size]
    return img

def load_and_resize(filepath):
    hdulist = fits.open(filepath)
    raw_image = hdulist[0].data
    if( raw_image is None ):
        raw_image = hdulist[1].data
    image = raw_image
    #image = zoom_img(image, raw_size[0], input_shape[0])
    trimmed_image = zoom_img(image, raw_size[0], input_shape[0])
    return (image, trimmed_image)

def to_png_and_save(fits_paths):
    output_paths = []
    count = 0
    raw_image_path = None
    for filepath in fits_paths:
        (image, trimmed_image) = load_and_resize(filepath)
        output_filepath = filepath.split('.')[0] + ".png"
        save_as_image(trimmed_image, output_filepath)
        output_paths.append(output_filepath)
        if count == 1:
            raw_image_path = filepath.split('.')[0] + "raw.png"
            save_as_image(image, raw_image_path)
        count = count + 1
    return (output_paths, raw_image_path)

for i, row in enumerate(reader):
    cat_id = row[0]
    paths = to_list(row[1])
    img_paths = []
    for path in paths:
        print(path)
        #replaced = path.replace('/disk/cos/ono', '/Users/daiz/disk/cos/ono')
        path = FILE_HOME + path
        img_paths.append(path)
    (png_img_paths, raw_image_path) = to_png_and_save(img_paths)
    img_tds = ''.join([make_img_td(filepath) for filepath in png_img_paths])
    combined_img_path = row[2].replace('/home/daiz', '/Users/daiz')
    label = row[3]
    #probabilities = [row[4], row[5]]
    probabilities = row[4][2:-2].split("', '")
    prob_tds = ''.join(['<td>%s</td>' % prob for prob in probabilities])
    answer = probabilities.index(max(probabilities))

    if answer == int(label):
        color = "#2EFE64"
    else:
        color = "#F78181"

    f_write.write('\t\t<tr bgcolor="%s">\n' % color)
    f_write.write("\t\t\t<td>%s</td>\n" % cat_id)
    f_write.write('\t\t\t%s\n' % make_img_td(raw_image_path))
    f_write.write('\t\t\t%s\n' % img_tds)
    f_write.write('\t\t\t%s\n' % make_img_td(combined_img_path))
    f_write.write("\t\t\t<td>%s</td>\n" % label)
    f_write.write('\t\t\t%s\n' % prob_tds)
    f_write.write('\t\t\t<td>%s</td>\n' % answer)
    f_write.write("\t\t</tr>\n")

    print("No. %s finished" % i)

f_write.write("\t</table>\n")
f_write.write("</html>\n")
