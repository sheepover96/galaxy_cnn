import csv

result_file = open('result.csv', 'w')
writer = csv.writer(result_file)

with open('dataset/cat_all.data', 'r') as f:
    reader = csv.reader(f, delimiter=' ')

    for row in reader:
        if row[3] == '2':
            row[3] = '0'
        writer.writerow(row)
    
