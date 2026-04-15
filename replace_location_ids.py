import csv
import shutil
from pathlib import Path

csv_path = Path('smart_traffic_management_dataset.csv')
backup_path = Path('smart_traffic_management_dataset.csv.bak')

if not csv_path.exists():
    raise SystemExit(f"CSV not found: {csv_path}")

# Mapping for location_id -> "City, Area"
mapping = {
    '1': 'Delhi, Connaught Place',
    '2': 'Mumbai, Andheri',
    '3': 'Bengaluru, Whitefield',
    '4': 'Ahmedabad, Gota',
    '5': 'Chennai, T. Nagar',
    '6': 'Kolkata, Salt Lake',
    '7': 'Hyderabad, Banjara Hills',
    '8': 'Pune, Koregaon Park',
    '9': 'Surat, Varachha',
    '10': 'Jaipur, MI Road',
    '11': 'Lucknow, Hazratganj',
    '12': 'Kanpur, Swaroop Nagar',
    '13': 'Nagpur, Sitabuldi',
    '14': 'Indore, Rajwada',
    '15': 'Bhopal, New Market'
}

# Backup original file
shutil.copy2(csv_path, backup_path)
print(f"Backup written to {backup_path}")

# Read and write
rows = []
with csv_path.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    if 'location_id' not in fieldnames:
        raise SystemExit('location_id column not found in CSV')
    for r in reader:
        lid = r['location_id']
        if lid in mapping:
            r['location_id'] = mapping[lid]
        else:
            # if numeric, convert to City_<id>
            if lid.isdigit():
                r['location_id'] = f"City_{lid}"
            # otherwise leave as is
        rows.append(r)

with csv_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {csv_path} with descriptive location names.")
print("Note: original file backed up to smart_traffic_management_dataset.csv.bak")
