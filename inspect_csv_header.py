import csv
with open('smart_traffic_management_dataset.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    print('HEADER:', header)
    print('LEN:', len(header))
