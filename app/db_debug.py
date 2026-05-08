import sqlite3
import sqlite_vec
import struct
from sentence_transformers import SentenceTransformer

# 1. Setup
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
db = sqlite3.connect("app/data/medical_data.db")
db.enable_load_extension(True)
sqlite_vec.load(db)

# 2. Test Vector Search
query = "Symptoms of Measles"
query_vector = embed_model.encode(query).tolist()
query_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)

print("--- Step 1: Searching Vectors ---")
# Using the STRICT syntax required by newer sqlite-vec
cursor = db.execute("""
    SELECT rowid, distance 
    FROM vec_medquad 
    WHERE embedding MATCH ? AND k = 5
    ORDER BY distance 
""", [query_bytes])
vec_results = cursor.fetchall()

if not vec_results:
    print("❌ FAIL: No vectors found. Your vector table is empty or the model is different.")
else:
    for rowid, dist in vec_results:
        print(f"✅ FOUND: Vector RowID {rowid} (Distance: {dist})")

        print("--- Step 2: Testing Join ---")
        res = db.execute("SELECT answer FROM medquad WHERE id = ?", [rowid]).fetchone()
        if res:
            print(f"🎯 SUCCESS! Found Text: {res[0][:50]}...")
        else:
            print(f"❌ FAIL: Vector RowID {rowid} exists, but ID {rowid} is MISSING in medquad table.")
            # Check what IDs actually exist
            sample = db.execute("SELECT id FROM medquad LIMIT 1").fetchone()
            print(f"   Diagnostic: The first ID in your text table is actually: {sample[0] if sample else 'NONE'}")

db.close()