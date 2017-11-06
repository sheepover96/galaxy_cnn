import os
import sys
import csv

argv = sys.argv

input_path = argv[1]
out_file_path = argv[2]

f_write = open(out_file_path, "w")
writer = csv.writer(f_write, lineterminator='\n')

#home_dir = '/Users/daiz/disk/cos/ono/HSC/S16A/for_deep_learning/cut_coadds/gri_part1/coadds/'

dir_list = os.listdir(input_path)
for d in dir_list:
    if d == '.DS_Store':
        continue
    cat_id = d.split('_')[1]
    fits_paths = [input_path+d+'/'+str(x)+'.fits' for x in range(1, 6)]
    attributes = [cat_id] + fits_paths
    print(attributes)
    writer.writerow(attributes)