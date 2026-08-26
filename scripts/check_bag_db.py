from pathlib import Path
import sqlite3


BAG_DIR = Path("test_session/bag")


def main():
    db_files = list(BAG_DIR.glob("*.db3"))

    if not db_files:
        print(f"No .db3 files found in {BAG_DIR}")
        return

    for db_file in db_files:
        print(f"\nChecking: {db_file}")
        print(f"Size: {db_file.stat().st_size / (1024 * 1024):.2f} MB")

        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            cursor.execute("PRAGMA integrity_check;")
            result = cursor.fetchone()

            print("Integrity check:", result[0])

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            print("Tables:")
            for table in tables:
                print(" ", table[0])

            conn.close()

        except Exception as e:
            print("ERROR:")
            print(e)


if __name__ == "__main__":
    main()