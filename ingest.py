import json
import os
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "rag.db"
DOCS_DIR = "docs"
CACHE_DIR = "models"   # modeller proje içindeki models/ klasörüne iner

def load_embedding_model(manager):
    catalog = manager.catalog
    emb_id = None
    for m in catalog.list_models():
        mid = getattr(m, "id", "")
        if "embedding" in mid and "cpu" in mid:
            emb_id = mid
            break
    if emb_id is None:
        raise RuntimeError("CPU embedding sürümü bulunamadı.")
    model = catalog.get_model_variant(emb_id)
    model.download(lambda p: print(f"\rModel iniyor: {p:.0f}%", end="", flush=True))
    print()
    model.load()
    return model

def read_documents(docs_dir):
    if not os.path.isdir(docs_dir):
        raise RuntimeError(f"'{docs_dir}' klasörü yok. Önce oluştur.")
    items = []
    for name in sorted(os.listdir(docs_dir)):
        if name.lower().endswith((".txt", ".md")):
            with open(os.path.join(docs_dir, name), "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                items.append((name, text))
    return items

def chunk_text(text, chunk_size=120, overlap=20):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks

def main():
    docs = read_documents(DOCS_DIR)
    if not docs:
        print(f"'{DOCS_DIR}' klasöründe .txt/.md dosyası yok.")
        return
    print(f"{len(docs)} dosya bulundu.")

    records = []
    for source, text in docs:
        for ch in chunk_text(text):
            records.append((source, ch))
    print(f"Toplam {len(records)} parça oluştu.")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS documents")
    cur.execute("""
        CREATE TABLE documents (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            source    TEXT NOT NULL,
            content   TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)

    config = Configuration(app_name="rag_local", model_cache_dir=CACHE_DIR)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    model = load_embedding_model(manager)
    print("Embedding modeli hazır.\n")

    client = model.get_embedding_client()
    texts = [content for _, content in records]
    vectors = []
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        resp = client.generate_embeddings(texts[i:i + BATCH])
        vectors.extend(item.embedding for item in resp.data)
        print(f"\r  embedding: {min(i + BATCH, len(texts))}/{len(texts)}", end="", flush=True)
    print()

    for (source, content), vec in zip(records, vectors):
        cur.execute(
            "INSERT INTO documents (source, content, embedding) VALUES (?, ?, ?)",
            (source, content, json.dumps(vec)),
        )
    conn.commit()
    model.unload()

    cur.execute("SELECT COUNT(*) FROM documents")
    print(f"{cur.fetchone()[0]} parça kaydedildi.")
    conn.close()

if __name__ == "__main__":
    main()