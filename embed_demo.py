import math
from foundry_local_sdk import Configuration, FoundryLocalManager

def cosine_similarity(a, b):
    """İki vektörün kosinüs benzerliği: 1'e yakın = çok benzer."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

def main():
    config = Configuration(app_name="rag_embed")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    catalog = manager.catalog

    # Embedding modelinin CPU sürümünü otomatik bul (CUDA bu makinede çalışmıyor)
    emb_id = None
    for m in catalog.list_models():
        mid = getattr(m, "id", "")
        if "embedding" in mid and "cpu" in mid:
            emb_id = mid
            break

    if emb_id is None:
        # Bulamazsak katalogdaki TÜM kimlikleri yazdır ki doğrusunu görelim
        print("CPU embedding sürümü bulunamadı. Katalogdaki modeller:")
        for m in catalog.list_models():
            print("   ", getattr(m, "id", m))
        return

    print(f"Embedding modeli: {emb_id}")
    model = catalog.get_model_variant(emb_id)
    model.download(lambda p: print(f"\rModel iniyor: {p:.0f}%", end="", flush=True))
    print()
    model.load()
    print("Embedding modeli hazır.\n")

    client = model.get_embedding_client()

    # Küçük bilgi tabanı — 4 farklı konuda cümle
    documents = [
        "Kediler bağımsız hayvanlardır, kendi başlarına vakit geçirmeyi severler.",
        "Python, yeni başlayanların kolayca öğrenebildiği popüler bir programlama dilidir.",
        "İstanbul Boğazı, Asya ile Avrupa kıtalarını birbirine bağlar.",
        "Köpekler sadık dostlardır ve sahiplerine çok düşkündür.",
    ]

    # Tüm belgeleri tek seferde (batch) vektöre çevir
    doc_resp = client.generate_embeddings(documents)
    doc_vecs = [item.embedding for item in doc_resp.data]

    # Soruyu vektöre çevir — bilerek belgelerden FARKLI kelimeler kullanıyorum
    query = "Yeni başlayanlar için hangi yazılım dilini öğrenmek mantıklı?"
    q_resp = client.generate_embedding(query)
    q_vec = q_resp.data[0].embedding

    # Her belgeyle benzerliği hesapla ve sırala
    print(f"Soru: {query}\n")
    scores = sorted(
        ((cosine_similarity(q_vec, vec), doc) for doc, vec in zip(documents, doc_vecs)),
        reverse=True,
    )
    print("Benzerlik sıralaması (yüksek = anlamca daha yakın):")
    for score, doc in scores:
        print(f"  {score:.3f}  {doc}")

    model.unload()

if __name__ == "__main__":
    main()