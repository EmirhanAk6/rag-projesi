from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    # SDK'yı başlat
    config = Configuration(app_name="rag_hello")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    catalog = manager.catalog

    # Takma ad yerine CPU sürümünü AÇIKÇA seç (CUDA bu makinede çalışmıyor)
    cpu_id = "qwen2.5-0.5b-instruct-generic-cpu:4"
    try:
        model = catalog.get_model_variant(cpu_id)
    except Exception:
        # Kimlik tutmazsa mevcut sürümleri yazdır ki doğrusunu görelim
        print("CPU sürümü bulunamadı. Mevcut qwen2.5-0.5b sürümleri:")
        for m in catalog.list_models():
            mid = getattr(m, "id", "")
            if "qwen2.5-0.5b" in mid:
                print("   ", mid)
        raise

    # İndir + yükle
    model.download(lambda p: print(f"\rModel iniyor: {p:.0f}%", end="", flush=True))
    print()
    model.load()
    print("Model hazır.\n")

    # Soru sor, cevabı token token yazdır
    client = model.get_chat_client()
    messages = [{"role": "user", "content": "Selam! Kısaca kendini tanıt."}]

    print("Asistan: ", end="", flush=True)
    for chunk in client.complete_streaming_chat(messages):
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    model.unload()

if __name__ == "__main__":
    main()