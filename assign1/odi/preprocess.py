import csv

with open('ODI-2018.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        print(row)
        print("hello")