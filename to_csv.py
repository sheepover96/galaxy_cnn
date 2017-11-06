import csv
import sys

args = sys.argv

input_filepath = args[1]
output_filepath = args[2]

result_file = open(output_filepath, 'w')
writer = csv.writer(result_file)

with open(input_filepath, 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in reader:
        #if row[3] == '2':
        #    row[3] = '0'
        writer.writerow(row)
    
