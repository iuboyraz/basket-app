# streamlit_app.py
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Başlık ve açıklama
st.title("Birliktelik Kuralları Analizi")
st.markdown("""
Bu uygulama, Apriori veya FP-Growth algoritmasını kullanarak birliktelik kurallarını analiz eder ve belirli bir ürün grubuna göre öneriler sunar.
""")

# Ürün listesi
urun_listesi = ["KARTUŞ", "OTOKLAV", "YIKAMA", "AMELİYAT MASASI", "REVERSE OSMOS", "HİDROJEN PEROKSİT", "OKSİJEN SİSTEMİ"]

# Dosya yükleme
uploaded_file = st.file_uploader("Lütfen Excel dosyasını yükleyin :", type=["xlsx"])

if uploaded_file is not None:
    # Veri yükleme
    df = pd.read_excel(uploaded_file)
    
    # Veri özetleri
    st.subheader("Veri Özeti")
    st.write(f"**Ürün Grubu Sayısı:** {df['urun_grubu'].nunique()}")
    st.write(f"**Toplam Sipariş:** {df['siparis_numarasi'].nunique()}")
    
    # One-Hot Encoding
    basket = df.pivot_table(index='siparis_numarasi', columns='urun_grubu', aggfunc='size', fill_value=0)
    basket = basket.map(lambda x: 1 if x > 0 else 0).astype(bool)
    
    # Algoritma seçimi
    algorithm = st.selectbox("Birliktelik Kuralları için Algoritma Seçin", options=["Apriori", "FP-Growth"])
    
    # min_support ve min_threshold değerlerini kullanıcıdan alma
    st.sidebar.header("Parametre Seçimleri")
    min_support = st.sidebar.slider("Min Support Değeri", min_value=0.01, max_value=1.0, value=0.01, step=0.01)
    min_threshold = st.sidebar.slider("Min Threshold Değeri", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    
    # Algoritma çalıştırma
    if algorithm == "Apriori":
        st.write("**Apriori Algoritması Çalıştırılıyor...**")
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    elif algorithm == "FP-Growth":
        st.write("**FP-Growth Algoritması Çalıştırılıyor...**")
        frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    
    # Sık ürün gruplarını sıralama
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True)
    st.write("**Sık Ürün Grupları:**")
    st.dataframe(frequent_itemsets)
    
    # Birliktelik kuralları
    st.write("**Birliktelik Kuralları Üretiliyor...**")
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_threshold, num_itemsets=len(frequent_itemsets))
    rules = rules.sort_values('confidence', ascending=False).reset_index(drop=True)
    st.write("**Birliktelik Kuralları:**")
    st.dataframe(rules)
    
    # Kullanıcının ürün seçimi
    urun_secimi = st.selectbox("Bir ürün grubu seçin", options=urun_listesi)
    
    # Seçilen ürün grubuna göre filtreleme
    if urun_secimi:
        st.subheader(f"Seçilen Ürün Grubu: {urun_secimi}")
        filtered_rules = rules[rules['antecedents'].apply(lambda x: urun_secimi in x)]
        
        if not filtered_rules.empty:
            st.write(f"**'{urun_secimi}' Ürünü için Önerilen Ürün/Ürün Grupları:**")
            st.dataframe(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        else:
            st.write(f"**'{urun_secimi}' ürünü ile ilgili kural bulunamadı.**")
else:
    st.info("Lütfen bir Excel dosyası yükleyin.")
