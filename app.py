import json
import math
import sqlite3
import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "rag.db"
CACHE_DIR = "models"
SIMILARITY_THRESHOLD = 0.30
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
            return mid
    return None

# Modeller bir kez yüklenir ve bellekte tutulur (her soruda tekrar yüklenmez)
@st.cache_resource(show_spinner="Modeller yükleniyor... (ilk açılışta biraz sürebilir)")
def load_models():
    config = Configuration(app_name="rag_local", model_cache_dir=CACHE_DIR)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    catalog = manager.catalog

    emb_model = catalog.get_model_variant(find_cpu_variant(catalog, ["embedding"]))
    emb_model.load()

    chat_id = find_cpu_variant(catalog, ["phi-4-mini", "instruct"])
    if chat_id is None:
        chat_id = find_cpu_variant(catalog, ["qwen2.5-0.5b"])
    chat_model = catalog.get_model_variant(chat_id)
    chat_model.load()

    return emb_model.get_embedding_client(), chat_model.get_chat_client()

def retrieve(query_vec, top_k=TOP_K):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT source, content, embedding FROM documents")
    scored = []
    for source, content, emb_json in cur.fetchall():
        vec = json.loads(emb_json)
        scored.append((cosine_similarity(query_vec, vec), source, content))
    conn.close()
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# --- Arayüz ---
st.set_page_config(page_title="Yerel RAG Asistanı", page_icon="📄")
st.title("📄 Yerel RAG Asistanı")
st.caption("Kendi belgelerinize dayanarak, çevrimdışı soru-cevap")

emb_client, chat_client = load_models()

question = st.text_input("Sorunuzu yazın:", placeholder="Örn: Foundry Local internet olmadan çalışır mı?")

if question:
    # 1. Soruyu vektöre çevir + en yakın parçaları getir
    q_vec = emb_client.generate_embedding(question).data[0].embedding
    top_chunks = retrieve(q_vec)

    # 2. Eşik kontrolü — uydurma engeli
    if not top_chunks or top_chunks[0][0] < SIMILARITY_THRESHOLD:
        st.warning("Bu konuda bilgim yok.")
    else:
        # 3. Bağlamı kur ve cevabı üret
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

        def stream_answer():
            for chunk in chat_client.complete_streaming_chat(messages):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        st.subheader("Cevap")
        st.write_stream(stream_answer)

        sources = sorted(set(s for _, s, _ in top_chunks))
        st.caption(f"Kaynak: {', '.join(sources)}")

        with st.expander("Getirilen parçalar (skorlarıyla)"):
            for score, source, content in top_chunks:
                st.markdown(f"**{score:.3f}** — *{source}*")
                st.text(content[:300])