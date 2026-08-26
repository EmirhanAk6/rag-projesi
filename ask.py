import json
import math
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "rag.db"
CACHE_DIR = "models"
SIMILARITY_THRESHOLD = 0.30   # bu skorun altındaysa LLM'e sorma, "bilgim yok" de
TOP_K = 3

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)

def find_cpu_variant(catalog, keywords):
    kws = [k.lower() for k in keywords]
    for m in catalog.list_models():
        mid = getattr(m, "id", "")
        low = mid.lower()
        if "cpu" in low and all(k in low for k in kws):
            return mid   # orijinal (büyük harfli) kimliği döndür
    return None

def load_variant(catalog, model_id):
    model = catalog.get_model_variant(model_id)
    model.download(lambda p: print(f"\r  {model_id} iniyor: {p:.0f}%", end="", flush=True))
    print()
    model.load()
    return model

def retrieve(cur, query_vec, top_k=TOP_K):
    """Tüm parçaları okuyup soruya en yakın top_k tanesini (skor, kaynak, metin) döndürür."""
    cur.execute("SELECT source, content, embedding FROM documents")
    scored = []
    for source, content, emb_json in cur.fetchall():
        vec = json.loads(emb_json)
        scored.append((cosine_similarity(query_vec, vec), source, content))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    config = Configuration(app_name="rag_local", model_cache_dir=CACHE_DIR)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    catalog = manager.catalog

    print("Modeller yükleniyor...")
    emb_model = load_variant(catalog, find_cpu_variant(catalog, ["embedding"]))
    chat_id = find_cpu_variant(catalog, ["phi-4-mini", "instruct"])
    if chat_id is None:
        chat_id = find_cpu_variant(catalog, ["qwen2.5-0.5b"])  # phi bulunamazsa eskisine dön
        print("Not: phi-4-mini bulunamadı, qwen2.5-0.5b kullanılıyor.")
    chat_model = load_variant(catalog, chat_id)
    emb_client = emb_model.get_embedding_client()
    chat_client = chat_model.get_chat_client()
    print("\nHazır! Soru sor ('çık' yazınca kapanır).\n")

    while True:
        question = input("Soru: ").strip()
        if question.lower() in ("çık", "cik", "quit", "exit", ""):
            break

        # 1. Soruyu vektöre çevir
        q_vec = emb_client.generate_embedding(question).data[0].embedding

        # 2. En yakın parçaları getir
        top_chunks = retrieve(cur, q_vec)

        print("  [getirilen parçalar]")
        for score, source, content in top_chunks:
            preview = content[:60].replace("\n", " ")
            print(f"    {score:.3f}  ({source})  {preview}...")

        # 3. Güvenlik eşiği — en yakın parça bile yeterince benzemiyorsa uydurma
        if not top_chunks or top_chunks[0][0] < SIMILARITY_THRESHOLD:
            print("Cevap: Bu konuda bilgim yok.\n")
            continue

        # 4. Bağlamı kur ve cevap üret
        context = "\n".join(f"- {c}" for _, _, c in top_chunks)
        system_msg = (
            "Sen bir soru-cevap asistanısın. SADECE aşağıdaki bağlamdaki bilgilere "
            "dayanarak Türkçe cevap ver. Bağlamda cevap yoksa 'Bu konuda bilgim yok.' de; "
            "kendi genel bilginden cevap uydurma.\n\n"
            f"Bağlam:\n{context}"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ]

        print("Cevap: ", end="", flush=True)
        for chunk in chat_client.complete_streaming_chat(messages):
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()

        # 5. Kaynağı göster
        sources = sorted(set(s for _, s, _ in top_chunks))
        print(f"  [kaynak: {', '.join(sources)}]\n")

    emb_model.unload()
    chat_model.unload()
    conn.close()
    print("Görüşürüz!")

if __name__ == "__main__":
    main()