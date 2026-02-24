import sqlite3

DB_PATH = "predictions.db"

# Create table
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        monsoon REAL,
        urbanization REAL,
        climate REAL,
        flood_risk INTEGER,
        probability REAL
    )
    """)

    conn.commit()
    conn.close()

# Insert prediction
def save_prediction(monsoon, urbanization, climate, flood_risk, probability):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (monsoon, urbanization, climate, flood_risk, probability)
    VALUES (?, ?, ?, ?, ?)
    """, (monsoon, urbanization, climate, flood_risk, probability))

    conn.commit()
    conn.close()

# Initialize database
init_db()


def get_predictions():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions")

    rows = cursor.fetchall()

    conn.close()

    result = []

    for row in rows:
        result.append({
            "id": row[0],
            "monsoon": row[1],
            "urbanization": row[2],
            "climate": row[3],
            "flood_risk": row[4],
            "probability": row[5]
        })

    return result