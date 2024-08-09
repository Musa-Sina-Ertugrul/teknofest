import sqlite3
import pandas as pd
from openpyxl import load_workbook
import os

def read_from_database(db_file, batch_size=1000):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM reviews")
    total_reviews = cursor.fetchone()[0]
    try:
        for offset in range(0, total_reviews, batch_size):
            cursor.execute(f"SELECT generated_text, output_json FROM reviews LIMIT {batch_size} OFFSET {offset}")
            yield cursor.fetchall()
    finally:
        conn.close()

def save_to_excel(df, excel_file):
    if os.path.exists(excel_file):
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            start_row = writer.sheets['Sheet1'].max_row if 'Sheet1' in writer.sheets else 0
            df.to_excel(writer, sheet_name='Sheet1', index=False, header=not bool(start_row), startrow=start_row)
    else:
        df.to_excel(excel_file, sheet_name='Sheet1', index=False)

def write_db_to_excel(db_file, excel_file, batch_size=1000):
    total_written = 0
    for batch in read_from_database(db_file, batch_size):
        df = pd.DataFrame(batch, columns=['Generated Text', 'Output JSON'])
        save_to_excel(df, excel_file)
        total_written += len(batch)
        print(f"{total_written} reviews written to Excel...")

if __name__ == "__main__":
    db_file = "/home/musasina/Desktop/projects/teknofest/turkish_company_reviews.db"
    excel_file = "turkish_company_reviews.xlsx"
    write_db_to_excel(db_file, excel_file)