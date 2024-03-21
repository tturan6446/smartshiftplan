import pandas as pd
import numpy as np
import boto3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Dosya yolunu tam olarak belirtin
dosya_yolu = r'C:\Users\tolga.turan\OneDrive - DİVAN TURİZM İŞLETMELERİ A.Ş\Desktop\Data1 Shift Schedule\dataworker.xlsx'

# Excel dosyasını okuyun
veri = pd.read_excel(dosya_yolu)
# BUSINESSDATE sütununu tarih formatına çevir ve yeni zaman serisi bileşenlerini çıkar
veri['BUSINESSDATE'] = pd.to_datetime(veri['BUSINESSDATE'])
veri['Year'] = veri['BUSINESSDATE'].dt.year
veri['Month'] = veri['BUSINESSDATE'].dt.month
veri['DayOfMonth'] = veri['BUSINESSDATE'].dt.day

# Haftanın gününü sayısal bir değişkene dönüştür
days = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
veri['WeekdayNum'] = veri['Day'].map(days)

# Ortalama sıcaklık hesapla
veri['AvgTemp'] = (veri['MaxTemp'] + veri['MinTemp']) / 2

# Yemek Sayısı/Çalışan Sayısı oranını hesapla
veri['YemekSayisi_CalisanSayisi'] = veri['Yemek Sayısı'] / veri['Çalışan Sayısı']
veri['YemekSayisi_CalisanSayisi'].replace([np.inf, -np.inf], np.nan, inplace=True)  # Sonsuz değerleri NaN ile değiştir
veri['YemekSayisi_CalisanSayisi'].fillna(0, inplace=True)  # NaN değerleri 0 ile doldur

# LOCATIONNAME ve diğer kategorik sütunları sayısal forma dönüştür (One-Hot Encoding)
veri = pd.get_dummies(veri, columns=['LOCATIONNAME',  'Explain'], drop_first=True)

# 'Day' sütununu label encoding uygula
label_encoder = LabelEncoder()
veri['Day'] = label_encoder.fit_transform(veri['Day'])
veri['TIMEPRD'] = label_encoder.fit_transform(veri['TIMEPRD'])

# İhtiyaç olmayan sütunları düşür
veri = veri.drop(['BUSINESSDATE'], axis=1)

# Bağımlı ve bağımsız değişkenleri belirle
X = veri.drop(['Çalışan Sayısı'], axis=1)  # Çalışan Sayısı bağımlı değişken dışında tüm sütunlar
y = veri['Çalışan Sayısı']

# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tahminleri saklamak için boş bir DataFrame oluştur
all_predictions_day_time_based = pd.DataFrame(index=X_test.index)

# Modelleri oluştur ve eğit
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "AdaBoost": AdaBoostRegressor(),
    "KNN": KNeighborsRegressor()
}

best_model = None
best_mse = float('inf')
best_r2 = -float('inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    diff = y_test - predictions
    print(f"{name} - MSE: {mse}, R^2: {r2}, Mean Difference: {diff.mean()}")

    if mse < best_mse:
        best_model = name
        best_mse = mse
        best_r2 = r2

# En iyi modeli yazdır
print(f"En iyi model: {best_model} - MSE: {best_mse}, R^2: {best_r2}")

# Tahminleri DataFrame'e ekle
all_predictions_day_time_based[best_model] = model.predict(X_test)

# Gerçek değerleri de DataFrame'e ekle
all_predictions_day_time_based['Gerçek Değerler'] = y_test.values
all_predictions_day_time_based['Day'] = X_test['Day'].map({0:'Pazartesi', 1:'Salı', 2:'Çarşamba', 3:'Perşembe', 4:'Cuma', 5:'Cumartesi', 6:'Pazar'}).values
all_predictions_day_time_based['TIMEPRD'] = X_test['TIMEPRD'].values

# Gün ve zaman dilimi bazında tahminleri grupla ve ortalama al
all_predictions_day_time_based = all_predictions_day_time_based.groupby(['Day', 'TIMEPRD']).mean().reset_index()

# Tahminleri Excel dosyasına yaz
all_predictions_day_time_based.to_excel('7_gunluk.xlsx', index=False)


#Day tanımlamasında 0 yok,0 çıktısı var ?
#Burada geleceğe dönük bir tahmin mi çalışıyor teyit edilmeli ?
#MSE değerleri olmalı
#MSE düşük
#R Kare 1'e ne kadar yakınsa
#Buradaki ne yapıldıysa teker teker açıklayan döküman hazırlanmalı
#Araştırma kaynakların ile, referans alınan yerler ile notlara yazılmalı

# VERİLERİ YUVARLAR VE PIVOT PLAN ÇIKARIR #
#Burada random forest dışına olursa patlar

# Excel dosyasını yükleme
df = pd.read_excel('7_gunluk.xlsx')

# "Gerçek Değerler" ve "Random Forest" sütunundaki değerleri 0 dijitli hale getirme
df['Gerçek Değerler'] = df['Gerçek Değerler'].round(0)
df['Random Forest'] = df['Random Forest'].round(0)

# Günleri özel bir sıraya göre sıralamak için kategorik veri tipi kullanma
days_order = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
df['Day'] = pd.Categorical(df['Day'], categories=days_order, ordered=True)

# Pivot tablosunu oluşturma
pivot_df = pd.pivot_table(df, values=['Random Forest'], index='Day', columns='TIMEPRD', aggfunc='sum')

# İşlenmiş veriyi yeni bir Excel dosyasına kaydetme
pivot_df.to_excel('7_gunluk_vardiya_plani.xlsx')

## AWS YÜKLEMESİ OLMAZSA BURAYI SİL ###

# AWS S3'e yükleme yapmak için boto3 kütüphanesi ile S3 client oluştur
s3_client = boto3.client('s3')

# Yüklenecek dosyanın yolu ve S3 bucket bilgileri
dosya_yolu = '7_gunluk_vardiya_plani.xlsx'
bucket_ismi = 'testssp'
s3_dosya_ismi = '7_gunluk_vardiya_plani.xlsx'  # S3'teki dosyanın adı

# Dosyayı S3'e yükle
try:
    s3_client.upload_file(dosya_yolu, bucket_ismi, s3_dosya_ismi)
    print(f"'{dosya_yolu}' dosyası '{bucket_ismi}' bucket'ına başarıyla yüklendi.")
except Exception as e:
    print(f"Dosya yükleme sırasında bir hata oluştu: {e}") 