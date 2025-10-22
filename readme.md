# ⚡ RAG TEMELLİ CHATBOT PROJESİ: ELEKTRİK VE ELEKTRONİK DERS NOTLARI ASİSTANI

> Bu depo, LangChain tabanlı bir RAG chatbot uygulamasıdır. Elektrik ve Elektronik ders notlarına dayalı olarak çalışır.

## 🎯 Projenin Amacı (Gereksinim 1)

Bu proje, **Retrieval-Augmented Generation (RAG)** mimarisini kullanarak, harici bir **Elektrik ve Elektronik** ders notları veri setine dayalı, bilgiye kapalı (grounded) bir soru-cevap asistanı geliştirmeyi amaçlamaktadır. Projenin temel hedefi, yalnızca sağlanan ders materyallerinin içeriğiyle sınırlı, **tutarlı, doğru ve akademik** cevaplar üretebilen bir chatbot sunmaktır.

## 📚 Veri Seti Hakkında Bilgi (Gereksinim 2)

Bu RAG sisteminde kullanılan bilgi kaynağı, **Analog Elektronik** ders notlarından oluşan bir PDF dosyasıdır. (İleride kolayca diğer ders notları eklenebilecek şekilde genel bir mimariyle tasarlanmıştır.)

* **Veri Tipi:** Teknik ders notu (PDF).

* **İçerik:** Diyotlar, transistörler, temel devre analizi ve pasif elektronik bileşenler gibi Analog Elektronik temel konularını kapsamaktadır.

* **Hazırlık Metodolojisi:** Veri seti, **`PyPDFLoader`** kullanılarak okunmuş ve **`RecursiveCharacterTextSplitter`** ile anlamlı metin parçalarına (chunk) ayrılmıştır. Bu parçalar daha sonra vektörlere çevrilmiştir.

## 🛠️ Çözüm Mimarisi ve Kullanılan Teknolojiler (Gereksinim 4)

Proje, **LangChain** çatısı etrafında kurulmuş bir **RAG mimarisini** kullanır. Bu mimari, LLM'in genel bilgi yerine ders notlarına odaklanmasını sağlayarak "halüsinasyon" riskini ortadan kaldırır.

| Bileşen Adı | Kullanılan Teknoloji | Görev | 
 | ----- | ----- | ----- | 
| **Büyük Dil Modeli (LLM)** | **OpenAI GPT-4o-mini** | Çekilen kaynak metinleri yorumlayarak nihai cevabı üretir. | 
| **Vektörleştirme (Embedding)** | **OpenAI Embeddings** | Metin parçalarını ve kullanıcı sorgusunu sayısal vektörlere çevirir. | 
| **Vektör Veritabanı** | **ChromaDB** | Vektörleri depolar ve sorgu anında en alakalı $k=3$ metin parçasını çeker (Retrieval). | 
| **Sorgulama Zinciri (RAG Chain)** | **LangChain's RetrievalQA** | Sorgulama, çekme (Retrieval) ve cevap üretme (Generation) süreçlerini yöneten ana beynidir. | 
| **Web Arayüzü** | **Streamlit** | Kullanıcı etkileşimini sağlayan basit, hızlı ve temiz arayüzü sunar. | 

## ⚙️ KODUN ÇALIŞMA KILAVUZU (Gereksinim 3) - **Streamlit Cloud Yayınlama Kılavuzu**

Bu proje, `app.py` dosyasında yaptığımız değişiklik nedeniyle artık **Streamlit Cloud** ortamına göre ayarlanmıştır. API anahtarının yönetimi için güvenli **`st.secrets`** yapısını kullanır.

### 1. GitHub Dosya Hazırlığı

Uygulamanın Streamlit Cloud'da çalışması için, yeni deponuzda (`rag-chatbot-proje`) aşağıdaki **4 ana bileşen** bulunmalıdır:

* **`app.py`**: İçinde **`st.secrets`** ile API anahtarını okuyan en son kodunuz.

* **`requirements.txt`**: Gerekli tüm Python bağımlılıklarını içeren dosya.

* **`chroma_db`**: **Önceden oluşturulmuş** vektör veritabanını içeren klasör.

* **`README.md`**: Bu belgenin kendisi.

### 2. Kütüphane Kurulumu (requirements.txt İçeriği)

Streamlit Cloud, bu dosyayı okuyarak gerekli tüm kütüphaneleri otomatik olarak kurar.


```txt
streamlit
langchain-core
langchain-community
langchain-openai
openai
tiktoken
chromadb
pypdf
pypdfium2
python-dotenv
faiss-cpu
```



### 3. API Anahtarı Ayarı (**Streamlit Secrets** - KRİTİK ADIM)

Kod, API anahtarınızı güvenli bir şekilde Streamlit Secrets üzerinden okuyacaktır. Bu ayar, **yalnızca Streamlit Cloud arayüzünde** yapılmalıdır. Yerel `.env` dosyası KULLANILMAZ.

1. Streamlit Cloud'da uygulamanızı yayınlarken veya Ayarlar (Settings) bölümünden **"Manage app"** menüsüne gidin.

2. **"Settings"** -> **"Secrets"** bölümünü açın.

3. Aşağıdaki formatta bir gizli anahtar ekleyin:



secrets.toml dosyasına eklenmesi gereken içerik

OPENAI_API_KEY = "sk-SENİN_ANAHTARIN_BURAYA_GELMELİ"


* **Uyarı:** Anahtar adı (`OPENAI_API_KEY`), `app.py` dosyasındaki kod ile birebir eşleşmelidir.

### 4. Vektör Veritabanının Oluşturulması (Initial Setup)
1.  Kullanmak istediğiniz PDF'leri **`data`** klasörüne yerleştirin.
2.  Ana Python dosyasını (`app.py`) çalıştırarak **`chroma_db`** klasörünü oluşturun:
    ```bash
    python app.py
    ```
    *Bu adım, LLM modelini kullanır ve biraz zaman alabilir.*

### 5. Chatbot'un Çalıştırılması
1.  Veritabanı oluştuktan sonra, Streamlit arayüzünü başlatın:
    ```bash
    streamlit run app.py
    ```

---

### 6. Chatbot'un Yayınlanması (Deploy)

1. Streamlit Cloud'da, yeni GitHub deponuzu (`rag-chatbot-proje`) seçin.

2. Gerekli ayarları (Branch, Main file path) kontrol edin ve **"Deploy!"** butonuna tıklayın.

## 🌐 Web Arayüzü ve Product Kılavuzu (Gereksinim 5)

Uygulama, temiz ve odaklanmış bir Streamlit arayüzü ile sunulmaktadır. Sayfanın başlığı ve simgesi, projenin **Elektrik ve Elektronik** temasını yansıtır.

### Çalışma Akışı

1. Arayüz, tarayıcıda açılır. Kullanıcı, sayfanın altındaki metin kutusuna ders notlarıyla ilgili sorusunu yazar.

2. Sistem, anlık olarak:
a. Kullanıcı sorusuna en alakalı 3 metin parçasını veritabanından çeker.
b. Bu parçaları ve soruyu **GPT-4o mini** modeline gönderir.
c. LLM tarafından üretilen cevabı ekrana basar.

### Test Önerisi

* **Test Sorusu 1** "Diyot nedir?"
* **Cevap:** " Asistan: Diyot, elektrik akımını yalnızca bir yönde ileten bir yarı iletken elemandır. İki terminale (anot ve katot) sahip olan diyot, ileri yönde kutuplandığında akım geçirebilirken, ters yönde kutuplandığında akım geçirmez. Diyotlar, çeşitli uygulamalarda kullanılır ve farklı türleri bulunmaktadır, örneğin, zener diyot, tünel diyot ve ışık yayan diyot (LED) gibi. Her bir diyot türü, belirli özellikleri ve çalışma prensipleri ile farklı amaçlar için tasarlanmıştır."

* **Test Sorusu 2** “MOSFET nasıl çalışır?”
* **Cevap:** " Asistan: MOSFET (Metal Oksit Yarı İletken FET), kapı ile kanal arasındaki SiO2 tabakası sayesinde izole bir yapıya sahiptir. MOSFET'in çalışma prensibi, kapıya uygulanan gerilime bağlı olarak değişir.

DE MOSFET'te, kapıya negatif gerilim uygulanırsa "Azaltma" (Depletion) modunda, pozitif gerilim uygulanırsa "Arttırma" (Enhancement) modunda çalışır. N kanal DE MOSFET'te, kapıya uygulanan gerilim belirli bir eşik değerine ulaştığında, kanalın iletkenliği artar ve akım geçişine izin verir.

E MOSFET'te ise fiziksel bir kanal yoktur. Kapıya uygulanan gerilim eşik değerinde olduğunda, SiO2 tabakasına komşu olan P malzemesinde ince bir negatif yük tabakası oluşur ve bu da bir kanal oluşturur. Eşik geriliminin altında bir kanal oluşmaz. Pozitif kapı kaynak gerilimi ile çalışır ve kapı kaynak gerilimi arttıkça kanaldaki elektron sayısı artar, böylece iletkenlik artar.

Özetle, MOSFET'ler kapıya uygulanan gerilim ile kontrol edilen yarı iletken elemanlardır ve bu gerilim değişiklikleri, kanalın iletkenliğini etkileyerek akım geçişini sağlar."

* **Test Sorusu 3** “Diyodun V-I Karakteristiği”
* **Cevap:** "Asistan: Diyodun V-I karakteristiği, diyodun akım (I) ve gerilim (V) arasındaki ilişkiyi gösteren bir grafiktir. Bu karakteristik, diyodun ileri ve ters yöndeki davranışını belirler. İleri yönde kutuplandığında, diyot belirli bir eşik gerilimine (örneğin, germanyum için 0.3 V ve silisyum için 0.6 V) ulaştığında akım geçirmeye başlar. Ters yönde ise diyot, belirli bir ters dayanma gerilimine kadar akım geçirmez. V-I karakteristiği, bu iki durumda diyodun akım ve gerilim değerlerini gösteren bir grafik olarak çizilir."

* **Test Sorusu 4** “Kovalent bağ nedir?”
* **Cevap:** "Asistan: Kovalent bağ, atomlar arasında elektronların ortak kullanılması ile oluşan bir bağdır. Aynı cins atomlar kovalent bağ ile üç boyutlu düzenli bir kristal yapı oluşturur. Bu bağ, paylaşılan her elektronun kendisini paylaşan iki komşu atomun çekirdeği tarafından eşit bir şekilde çekilmesi ile atomları bir arada tutar. Kovalent bağ ile elektron paylaşımı, atomun nötr olmasını değiştirmez."

* **Test Sorusu 5** “Atom bilgisi”
* **Cevap:** " Asistan: Bu bilgi belgeden çıkarılamıyor." "(Çıkaramadığı bilginin de bu şekilde yanıtlandığını görmüş oluyoruz."

## 📸 Ekran Görüntüleri | Görsel 1 | Görsel 2 | Görsel 3 | |:-:|:-:|:-:| | ![](assets/1.png) | ![](assets/2.png) | ![](assets/3.png) | | Görsel 4 | Görsel 5 | Görsel 6 | |:-:|:-:|:-:| | ![](assets/4.png) | ![](assets/5.png) | ![](assets/6.png) | ## 🎥 Demo Videosu Proje tanıtım videosu: [👉 İzlemek için tıklayın](assets/chatbotvideo.mp4)

*Görsel 6: Sistem, verilen soruya olumsuz yanıt verdiğini göstermektedir.*

### 🔗 Uygulama Linki (Deploy Linki Buraya Gelecek)

**Web Linki:** `https://eem-ders-asistani.streamlit.app/`

---

## 👤 Yazar

**Ayşe Nur Kar Uzun**  
[GitHub Profilim](https://github.com/aayse-nur) | [LinkedIn Profilim](https://www.linkedin.com/in/ayse-nur-kar/)



