import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / 'modelos' / 'models.db'
print('DB path:', DB_PATH, 'exists:', DB_PATH.exists())

if not DB_PATH.exists():
    raise SystemExit(0)

con = sqlite3.connect(str(DB_PATH))
cur = con.cursor()

print('Tables:')
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print([r[0] for r in cur.fetchall()])

def print_table_count(name: str):
    try:
        cur.execute(f"SELECT COUNT(*) FROM {name}")
        c = cur.fetchone()[0]
        print(f"{name} count:", c)
        return c
    except Exception as e:
        print(f"{name} error:", e)
        return None

print_table_count('modelos')
print_table_count('models')
print_table_count('slots')

print('\nTop modelos:')
try:
    cur.execute("SELECT id, nome, image_path, criado_em, atualizado_em FROM modelos ORDER BY atualizado_em DESC LIMIT 20")
    for row in cur.fetchall():
        print(row)
except Exception as e:
    print('list modelos error:', e)

con.close()


