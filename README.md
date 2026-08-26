# Yerel RAG Asistanı — Foundry Local

Kendi belgelerinize dayanarak soru-cevap yapan, **tamamen çevrimdışı** çalışan bir yapay zeka asistanı. Microsoft **Foundry Local** ile modelleri cihaz üzerinde çalıştırır; hiçbir veri buluta gönderilmez.

Asistan, sorulan soruya belgelerinizdeki en ilgili metin parçalarını bulur, bunları dil modeline bağlam olarak verir ve cevabı bu bağlama dayanarak üretir. Belgelerde cevap yoksa uydurmaz, "Bu konuda bilgim yok." der.

## Özellikler

- **Çevrimdışı ve gizli:** Tüm işlemler yerel makinede yapılır, internet gerekmez.
- **Kaynağa dayalı cevaplar:** Her cevabın hangi dosyadan geldiği gösterilir.
- **Uydurma engeli:** Benzerlik eşiğinin altındaki sorularda model cevap üretmez.
- **Kendi belgelerinizle çalışır:** `docs/` klasörüne koyduğunuz `.txt` / `.md` dosyalarını okur.

## Nasıl Çalışır?

Sistem, klasik **RAG (Retrieval-Augmented Generation)** akışını izler:

1. **Getirme (Retrieve):** Soru bir embedding modeliyle sayı vektörüne çevrilir; SQLite'taki belge parçalarıyla kosinüs benzerliği hesaplanarak en yakın parçalar bulunur.
2. **Zenginleştirme (Augment):** Bulunan parçalar, soruyla birlikte dil modeline bağlam olarak verilir.
3. **Üretme (Generate):** Model, yalnızca bu bağlama dayanarak cevabı üretir.

```
Soru → embedding → SQLite'ta benzerlik araması → en yakın parçalar
     → (bağlam + soru) → dil modeli → cevap + kaynak
```

Kullanılan modeller (ikisi de CPU üzerinde çalışır):

- **Embedding:** `qwen3-embedding-0.6b` — metni anlam vektörüne çevirir.
- **Sohbet (LLM):** `Phi-4-mini-instruct` — bağlama dayalı cevabı yazar.

## Gereksinimler

- Windows
- Python 3.11 veya üzeri
- Microsoft Foundry Local SDK
- Yeterli disk alanı (modeller ~3-4 GB yer kaplar)

## Kurulum

```powershell
# 1. Sanal ortam oluştur ve aktifleştir
py -m venv rag-env
rag-env\Scripts\activate

# 2. Gerekli paketleri kur
pip install foundry-local-sdk-winml openai
```

> Not: Bu proje modelleri **CPU** üzerinde çalıştırır. GPU (CUDA) hızlandırması bazı makinelerde uyumsuzluk verdiği için, kod modellerin CPU sürümlerini açıkça seçer.

## Kullanım

### 1. Belgeleri hazırla

Proje kökündeki `docs/` klasörüne kendi metin dosyalarınızı (`.txt` veya `.md`, UTF-8) koyun.

### 2. Belgeleri işle (bir kez)

```powershell
py ingest.py
```

Bu komut belgeleri okur, küçük parçalara böler, her parçayı embedding'e çevirir ve `rag.db` adlı SQLite veritabanına kaydeder. Belgeler değişmedikçe tekrar çalıştırmaya gerek yoktur.

### 3. Soru sor

```powershell
py ask.py
```

Modeller yüklendikten sonra soru sorabilirsiniz. Her cevabın altında kaynak dosya gösterilir. Çıkmak için `çık` yazın.

Örnek:

```
Soru: Foundry Local internet olmadan çalışır mı?
Cevap: Evet, Foundry Local tamamen yerel çalışır ve verileriniz cihazdan dışarı çıkmaz.
  [kaynak: ornek.txt]
```

## Proje Yapısı

```
rag-projesi/
├── docs/           # Kaynak belgeler (.txt / .md)
├── models/         # Foundry Local model önbelleği
├── ingest.py       # Belgeleri işleyip veritabanına yazar
├── ask.py          # Soru-cevap arayüzü
├── rag.db          # SQLite veritabanı (belgeler + embedding'ler)
└── README.md
```

## Teknik Detaylar ve Tasarım Kararları

- **Parçalama (chunking):** Belgeler ~120 kelimelik, 20 kelime örtüşen parçalara bölünür. Örtüşme, cümlelerin parça sınırında anlamını kaybetmesini önler.
- **Vektör saklama:** Embedding'ler SQLite'ta JSON metni olarak saklanır; benzerlik kosinüs benzerliğiyle hesaplanır.
- **Getirme:** Her soru için en yakın 3 parça (top_k=3) bağlam olarak kullanılır.
- **Uydurma engeli:** En yakın parçanın benzerlik skoru 0.30'un altındaysa, model hiç çağrılmadan "Bu konuda bilgim yok." döndürülür.
- **Model önbelleği:** `Configuration(model_cache_dir="models")` ile modeller proje içindeki `models/` klasörüne indirilir.

## Sınırlamalar

- Modeller CPU'da çalıştığı için cevaplar birkaç saniye gecikebilir.
- Cevap kalitesi seçilen dil modelinin boyutuna bağlıdır; daha büyük modeller daha akıcı cevap verir ama daha yavaştır.
- Basit doğrusal benzerlik araması kullanılır; çok büyük belge koleksiyonları için özel bir vektör indeksleme gerekebilir.
