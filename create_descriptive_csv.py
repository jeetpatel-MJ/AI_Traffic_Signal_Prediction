import csv
from pathlib import Path
import shutil

SRC = Path('smart_traffic_management_dataset.csv')
OUT = Path('smart_traffic_management_dataset_descriptive.csv')
BACKUP = Path('smart_traffic_management_dataset.csv.orig_bak')

MAPPING = {
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


def safe_create():
    if not SRC.exists():
        print(f"Source file not found: {SRC}")
        return
    size = SRC.stat().st_size
    if size < 200:
        print(f"Source file looks too small ({size} bytes). Aborting to avoid data loss.")
        return

    # Read CSV and check header
    with SRC.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or 'location_id' not in fieldnames:
            print("CSV header missing or doesn't contain 'location_id'. Aborting.")
            return

        rows = []
        for r in reader:
            lid = r.get('location_id')
            if lid is None:
                rows.append(r)
                continue
            sid = str(lid).strip()
            if sid in MAPPING:
                r['location_id'] = MAPPING[sid]
            rows.append(r)

    # Backup original (non-destructive)
    if not BACKUP.exists():
        shutil.copy2(SRC, BACKUP)
        print(f"Backup of original created at {BACKUP}")
    else:
        print(f"Backup already exists at {BACKUP}")

    # Write out descriptive CSV
    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Descriptive CSV written to {OUT}")


if __name__ == '__main__':
    safe_create()
