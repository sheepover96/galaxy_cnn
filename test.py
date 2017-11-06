import sys
import csv
from astropy.io import fits
import aplpy
from PIL import Image
import numpy as np

img_width = 100
img_height = 100

img_channels = 3
raw_size = (48, 48, img_channels)
input_shape = (24, 24, img_channels)

def to_list(string):
    return string[2:-2].split("', '")

def make_img_td(filepath):
    return '<td><img src="%s" width="%s" height="%s" title="%s"></td>' % (filepath, img_width, img_height, filepath)

def normalize(image):
    return (image - image.min()).astype(float)*255 / (image.max() - image.min()).astype(float)

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
    image = zoom_img(image, raw_size[0], input_shape[0])
    return image

def to_png_and_save(fits_paths):
    output_paths = []
    for filepath in fits_paths:
        image = load_and_resize(filepath)
        output_filepath = filepath.split('.')[0] + ".png"
        save_as_image(image, output_filepath)
        output_paths.append(output_filepath)
    return output_paths

#input_file = "/Users/daiz/summarize_170607/good/warps_76552814184124838_3/coaddfr270.fits" # NG 48 49
#input_file = "/Users/daiz/summarize_170607/good/warps_43163283158473900_3/coaddr270.fits" # OK 49 48
input_file = "/Users/daiz/summarize_170607/good/warps_41619014782325143_3/coaddr90.fits" # NG 49 49
hdulist = fits.open(input_file)
raw_image = hdulist[0].data
(rows, cols) = (raw_image.shape[0], raw_image.shape[1])
print("raw rows = %s, cols = %s" % (rows, cols))
save_as_image(raw_image, "./raw.png")

resized_image = np.resize(raw_image, [raw_size[0], raw_size[1]])
(rows, cols) = (resized_image.shape[0], resized_image.shape[1])
print("resized rows = %s, cols = %s" % (rows, cols))
save_as_image(resized_image, "./resized.png")

image = load_and_resize(input_file)
save_as_image(image, "./hoge.png")

"""
for i, row in enumerate(reader):
    cat_id = row[0]
    img_paths = to_list(row[1])
    png_img_paths = to_png_and_save(img_paths)
    img_tds = ''.join([make_img_td(filepath) for filepath in png_img_paths])
    combined_img_path = row[2]
    label = row[3]
    probabilities = to_list(row[4])
    prob_tds = ''.join(['<td>%s</td>' % prob for prob in probabilities])
    answer = probabilities.index(max(probabilities))

    if answer == int(label):
        color = "#2EFE64"
    else:
        color = "#F78181"

    f_write.write('\t\t<tr bgcolor="%s">\n' % color)
    f_write.write("\t\t\t<td>%s</td>\n" % cat_id)
    f_write.write('\t\t\t%s\n' % img_tds)
    f_write.write('\t\t\t%s\n' % make_img_td(combined_img_path))
    f_write.write("\t\t\t<td>%s</td>\n" % label)
    f_write.write('\t\t\t%s\n' % prob_tds)
    f_write.write('\t\t\t<td>%s</td>\n' % answer)
    f_write.write("\t\t</tr>\n")

    print("No. %s finished" % i)

f_write.write("\t</table>\n")
f_write.write("</html>\n")
"""