import random
import json
from groq import Groq
import sqlite3
from datetime import datetime
import signal
import sys

# Constants (expanded)
COMPANIES_CATEGORIES = {
    "Telekomünikasyon": ["Turkcell", "Türk Telekom", "Vodafone Türkiye", "Turknet", "Superonline"],
    "E-ticaret": ["Trendyol", "Hepsiburada", "N11", "GittiGidiyor", "Çiçeksepeti", "Amazon Türkiye"],
    "Bankacılık": ["Ziraat Bankası", "İş Bankası", "Garanti BBVA", "Akbank", "Yapı Kredi", "QNB Finansbank"],
    "Havayolu": ["Türk Hava Yolları", "Pegasus", "AnadoluJet", "SunExpress", "Onur Air"],
    "Gıda": ["Ülker", "Eti", "Sek", "Pınar", "Torku", "Dimes"]
}

PRODUCTS_SERVICES = {
    "Telekomünikasyon": {
        "ürünler": ["fiber internet", "mobil hat", "ev telefonu", "TV+", "4.5G hizmetleri", "akıllı ev çözümleri"],
        "hizmetler": ["müşteri desteği", "online işlemler", "fatura ödeme", "tarife değişikliği", "yurtdışı paketleri"]
    },
    "E-ticaret": {
        "ürünler": ["elektronik", "giyim", "ev eşyaları", "kozmetik", "kitap", "spor malzemeleri"],
        "hizmetler": ["hızlı teslimat", "kolay iade", "indirim kuponları", "güvenli ödeme", "müşteri yorumları"]
    },
    "Bankacılık": {
        "ürünler": ["kredi kartı", "mevduat hesabı", "bireysel emeklilik", "konut kredisi", "araç kredisi", "yatırım fonu"],
        "hizmetler": ["mobil bankacılık", "internet bankacılığı", "ATM hizmetleri", "şube işlemleri", "7/24 müşteri hizmetleri"]
    },
    "Havayolu": {
        "ürünler": ["ekonomi sınıfı bilet", "business sınıfı bilet", "mil programları", "ekstra bagaj hakkı", "özel menü seçenekleri"],
        "hizmetler": ["online check-in", "uçuş değişikliği", "koltuk seçimi", "lounge erişimi", "transfer hizmetleri"]
    },
    "Gıda": {
        "ürünler": ["bisküvi", "süt", "çikolata", "meyve suyu", "dondurma", "yoğurt"],
        "hizmetler": ["online sipariş", "market teslimatı", "ürün çeşitliliği", "özel üretim", "kalite kontrol"]
    }
}

SENTIMENT_EXPRESSIONS = ['olumlu', 'olumsuz', 'nötr']

# New constants
AGE_RANGES = ["18-25", "26-35", "36-45", "46-55", "56+"]
GENDERS = ["Erkek", "Kadın", "Belirtilmemiş"]
LOCATIONS = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Trabzon", "Eskişehir", "Gaziantep"]
OCCUPATIONS = ["Öğrenci", "Memur", "Özel Sektör Çalışanı", "Emekli", "Ev Hanımı", "Serbest Meslek", "İşsiz"]
TARGET_DEMOGRAPHICS = ["Gençler", "Aileler", "Profesyoneller", "Öğrenciler", "Emekliler"]
PLATFORMS = ["Şirket Websitesi", "Ekşi Sözlük", "Instagram", "Twitter", "Facebook", "Google Reviews"]
DEVICES = ["Akıllı Telefon", "Tablet", "Bilgisayar"]
TIMES_OF_DAY = ["Sabah", "Öğlen", "Akşam", "Gece"]
DAYS_OF_WEEK = ["Hafta İçi", "Hafta Sonu"]
FORMALITY_LEVELS = ["Resmi", "Yarı Resmi", "Günlük", "Çok Samimi"]
EMOTIONS = ["Heyecanlı", "Hayal Kırıklığı", "Memnun", "Öfkeli", "Nötr"]
EMOJI_USAGE = ["Yok", "Az", "Orta", "Çok"]
SLANG_USAGE = ["Yok", "Az", "Orta", "Çok"]
LOYALTY_STATUSES = ["İlk Kez Kullanan", "Ara Sıra Kullanan", "Düzenli Müşteri", "Sadık Müşteri"]
PRICE_PERCEPTIONS = ["Ucuz", "Makul", "Pahalı", "Çok Pahalı"]
SEASONS = ["İlkbahar", "Yaz", "Sonbahar", "Kış"]
TECHNICAL_EXPERTISE = ["Acemi", "Orta Düzey", "İleri Düzey"]

POSITIONS = ["başlangıç", "orta", "son"]
MIN_WORDS = 10
MAX_WORDS = 25

GROQ_MODELS = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
]

def generate_company_data(num_companies=None):
    if num_companies is None:
        num_companies = random.randint(1, 3)
    
    category = random.choice(list(COMPANIES_CATEGORIES.keys()))
    companies = random.sample(COMPANIES_CATEGORIES[category], k=min(num_companies, len(COMPANIES_CATEGORIES[category])))
    
    companies_data = []
    for company in companies:
        product = random.choice(PRODUCTS_SERVICES[category]["ürünler"])
        service = random.choice(PRODUCTS_SERVICES[category]["hizmetler"])
        
        companies_data.append({
            "company": company,
            "category": category,
            "sentiment": random.choice(SENTIMENT_EXPRESSIONS),
            "position": random.choice(POSITIONS),
            "product": product,
            "service": service,
            "target_demographic": random.choice(TARGET_DEMOGRAPHICS)
        })
    
    return companies_data

def generate_user_data():
    return {
        "age_range": random.choice(AGE_RANGES),
        "gender": random.choice(GENDERS),
        "location": random.choice(LOCATIONS),
        "occupation": random.choice(OCCUPATIONS),
        "loyalty_status": random.choice(LOYALTY_STATUSES),
        "technical_expertise": random.choice(TECHNICAL_EXPERTISE)
    }

def generate_comment_context():
    return {
        "platform": random.choice(PLATFORMS),
        "device": random.choice(DEVICES),
        "time_of_day": random.choice(TIMES_OF_DAY),
        "day_of_week": random.choice(DAYS_OF_WEEK),
        "formality": random.choice(FORMALITY_LEVELS),
        "emotion": random.choice(EMOTIONS),
        "emoji_usage": random.choice(EMOJI_USAGE),
        "slang_usage": random.choice(SLANG_USAGE),
        "price_perception": random.choice(PRICE_PERCEPTIONS),
        "season": random.choice(SEASONS)
    }

def create_prompt(companies_data, user_data, comment_context):
    prompt = f"""Lütfen aşağıdaki Türk şirketleri hakkında gerçekçi bir müşteri yorumu oluşturun. Tüm şirketler {companies_data[0]['category']} kategorisindedir:

Kullanıcı Profili:
- Yaş Aralığı: {user_data['age_range']}
- Cinsiyet: {user_data['gender']}
- Konum: {user_data['location']}
- Meslek: {user_data['occupation']}
- Müşteri Sadakati: {user_data['loyalty_status']}
- Teknik Bilgi Seviyesi: {user_data['technical_expertise']}

Yorum Bağlamı:
- Platform: {comment_context['platform']}
- Cihaz: {comment_context['device']}
- Zaman: {comment_context['time_of_day']}, {comment_context['day_of_week']}
- Yazı Stili: {comment_context['formality']}
- Duygu Durumu: {comment_context['emotion']}
- Emoji Kullanımı: {comment_context['emoji_usage']}
- İnternet Jargonu Kullanımı: {comment_context['slang_usage']}
- Fiyat Algısı: {comment_context['price_perception']}
- Mevsim: {comment_context['season']}

Şirketler:
"""

    for company_data in companies_data:
        prompt += f"""
Şirket: {company_data['company']}
Duygu: {company_data['sentiment']}
Cümledeki konum: {company_data['position']}
Ürün: {company_data['product']}
Hizmet: {company_data['service']}
Hedef Kitle: {company_data['target_demographic']}
"""

    prompt += """
Yorum şu özelliklere sahip olmalıdır:
1. Her şirket için belirtilen duyguyu (olumlu, olumsuz veya nötr) deneyimini anlatarak ifade et, yaratıcı ol benliğini kaybetme.
2. Şirketlerin ürün veya hizmetlerinden bahsedebilirsin.
3. Şirketleri cümle içinde belirtilen konumlar neresiyse oralarda (başlangıç, orta, son) kullanın.
4. Yorumunuz doğal ve gerçekçi olmalı, sanki gerçek bir müşteri yani insan tarafından yazılmış gibi.
5. Yorum en az 10, en fazla 25 kelime olmalı. Yani kullanıcı yorumu kısa olsun ve hemen konuşmayı kes.
6. Verilen şirketler arası karşılaştırmalar yapabilir.
7. Kategori içi tutarlılığı koruyun, sadece verilen kategori için geçerli terimler ve deneyimler kullanın.
8. Kullanıcı profiline ve yorum bağlamına uygun bir dil ve ton kullanın.
9. Gerekirse fiyat algısını, mevsimsel etkileri veya teknik bilgi seviyesini yansıtın.
10. Kullanıcının sadakat durumuna uygun bir yorum yazın.
11. Platform ve cihaza özgü deneyimleri yansıtabilirsiniz.

Lütfen sadece müşteri yorumunu yaz, başka hiçbir ekleme istemiyorum.

Yorum kesinlikle ama kesinlikle Türkçe olmalı. Türk kullanıcı yorumu istiyorum."""

    return prompt

def generate_review(client, model, companies_data, user_data, comment_context):
    prompt = create_prompt(companies_data, user_data, comment_context)

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=500,
        temperature=0.5,
        top_p=1,
    )

    generated_review = chat_completion.choices[0].message.content.strip()
    return generated_review

def initialize_database(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS reviews
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       generated_text TEXT,
                       output_json TEXT,
                       timestamp DATETIME)''')
    conn.commit()
    return conn

def save_to_database(conn, generated_text, output_json):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO reviews (generated_text, output_json, timestamp) VALUES (?, ?, ?)",
                   (generated_text, output_json, datetime.now()))
    conn.commit()

def graceful_shutdown(signum, frame):
    print("\nReceived signal to shut down. Closing database connection...")
    if 'conn' in globals() and conn:
        conn.close()
    print("Database connection closed. Exiting...")
    sys.exit(0)

def generate_reviews_to_db(num_reviews, db_file):
    global conn
    client = Groq(api_key="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")
    conn = initialize_database(db_file)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    try:
        for i in range(num_reviews):
            model = random.choice(GROQ_MODELS)
            companies_data = generate_company_data()
            user_data = generate_user_data()
            comment_context = generate_comment_context()
            
            try:
                review = generate_review(client, model, companies_data, user_data, comment_context)
            except Exception as e:
                print(f"Error generating review: {e}")
                continue
            
            output = {
                "entity_list": [company_data["company"] for company_data in companies_data],
                "results": [
                    {
                        "entity": company_data["company"],
                        "sentiment": company_data["sentiment"]
                    }
                    for company_data in companies_data
                ],
                "user_data": user_data,
                "comment_context": comment_context
            }
            
            output_json = json.dumps(output, ensure_ascii=False)
            
            save_to_database(conn, review, output_json)
            
            if (i + 1) % 100 == 0:
                print(f"{i + 1} reviews generated and saved to database...")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

    print(f"{num_reviews} reviews have been generated and saved to {db_file}")

if __name__ == "__main__":
    db_file = "turkish_company_reviews.db"
    generate_reviews_to_db(10000, db_file)