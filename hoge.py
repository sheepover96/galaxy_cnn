import csv

f_read = open("dataset/strong_lae.csv", "r")
f_write = open("dataset/lae.csv", "w")

reader = csv.reader(f_read)

for row in reader:
    id = row[0].split("/")[-2]
    id = id.split("_")[1]
    row.insert(0, id)
    f_write.write(",".join(row)+"\n")
