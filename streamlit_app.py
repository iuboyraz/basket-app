import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from io import BytesIO

# Streamlit başlığı ve açıklama
st.title("Sepet Analizi")
st.write("Excel formatındaki sipariş verilerinizden sık kullanılan ürün gruplarını ve önerileri keşfedin!")

# Ürün listesi
urun_listesi = [
    "KARTUŞ", "OTOKLAV", "YIKAMA", "AMELİYAT MASASI", 
    "REVERSE OSMOS", "HİDROJEN PEROKSİT", "OKSİJEN SİSTEMİ"
]

# Dosya yükleme
uploaded_file = st.file_uploader("Lütfen bir Excel dosyası yükleyin:", type=["xlsx"])

# Algoritma seçimi
algorithm = st.selectbox("Sepet analizi algoritmasını seçin:", ["Apriori", "FP-Growth"])

# Minimum destek ve güven eşiklerinin ayarlanması
min_support = st.slider("Minimum Destek (Support):", min_value=0.01, max_value=1.0, value=0.01, step=0.01)
min_threshold = st.slider("Minimum Güven (Confidence):", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

# Ürün seçimi (çoklu seçim)
urun_secimi = st.multiselect("Analiz için ürün(ler) seçin:", urun_listesi)

# Filtreleme yöntemi seçimi
filter_method = st.selectbox(
    "Filtreleme yöntemini seçin:",
    [
        "1 - Seçilen ürün grubunun içinde bulunduğu sepetler değerlendirilerek öneride bulunulacaktır.",
        "2 - Seçilen ürün grubunun mutlak olarak içinde bulunduğu sepetler değerlendirilerek öneride bulunulacaktır.",
    ]
)

# Dosya yüklendiğinde analiz yapılır
if uploaded_file is not None:
    # Veriyi yükle
    df = pd.read_excel(BytesIO(uploaded_file.read()))

    # Veri özeti
    st.write("### Veri Özeti")
    st.write(f"- **Ürün Grubu Sayısı**: {df['urun_grubu'].nunique()}")
    st.write(f"- **Toplam Sipariş Sayısı**: {df['siparis_numarasi'].nunique()}")

    # One-Hot Encoding
    basket = df.pivot_table(index='siparis_numarasi', columns='urun_grubu', aggfunc='size', fill_value=0)
    basket = basket.map(lambda x: 1 if x > 0 else 0).astype(bool)

    # Sık ürün gruplarını bul
    if algorithm == "Apriori":
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)

    # Sık ürün gruplarını görüntüle
    st.write("### Sık Ürün Grupları")
    st.dataframe(frequent_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True))

    # Birliktelik kurallarını hesapla
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_threshold, num_itemsets=len(frequent_itemsets))
    rules = rules.sort_values('confidence', ascending=False).reset_index(drop=True)

    # Birliktelik kurallarını görüntüle
    st.write("### Birliktelik Kuralları")
    st.dataframe(rules)

    # Seçilen ürün grubuna göre filtreleme
    if urun_secimi:
        st.write(f"### Seçilen Ürün(ler): {', '.join(urun_secimi)}")
        selected_products = list(urun_secimi)

        if filter_method.startswith("1"):
            # Yöntem 1: Seçilen ürün(ler) grubunun içinde bulunduğu sepetler
            filtered_rules = rules[rules['antecedents'].apply(lambda x: any(product in x for product in selected_products))]
            st.write("#### Filtreleme Yöntemi 1: Seçilen ürün(ler) sepet içinde bulunduğu kurallar")
        else:
            # Yöntem 2: Seçilen ürün(ler) grubunun tam olarak bulunduğu sepetler
            filtered_rules = rules[rules['antecedents'].apply(lambda x: set(x) == set(selected_products))]
            st.write("#### Filtreleme Yöntemi 2: Seçilen ürün(ler) sepet içinde mutlak olarak bulunduğu kurallar")

        # Filtrelenmiş kuralları göster
        if not filtered_rules.empty:
            st.write("### Önerilen Ürün/Ürün Grupları")
            st.dataframe(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        else:
            st.warning("Seçilen ürün(ler) ile ilgili kural bulunamadı.")
    else:
        st.info("Lütfen analiz için en az bir ürün seçin.")
