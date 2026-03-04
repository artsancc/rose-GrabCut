# GrabCut ile Gül Rengi Değişimi

Bu Python projesi, bir görüntüdeki kırmızı gülü gelişmiş görüntü işleme teknikleri kullanarak tespit eder, arka plandan ayırır ve doğal bir geçişle rengini mora dönüştürür.
Sıradan filtrelerin aksine, bu çalışma GrabCut algoritmasını özel HSV renk maskeleriyle besleyerek çok daha hassas bir seçim alanı oluşturur.

# Özellikler

Çift Katmanlı Renk Tespiti: 
Kırmızı rengin HSV spektrumundaki iki farklı bölgesini de (başlangıç ve bitiş) yakalayarak tam kapsama sağlar.

GrabCut Entegrasyonu:
Dikdörtgen tabanlı (GC_INIT_WITH_RECT) ve maske tabanlı (GC_INIT_WITH_MASK) segmentasyonu birleştirerek nesne sınırlarını keskinleştirir.

Gürültü Temizleme:
Morfolojik işlemler ve connectedComponentsWithStats kullanarak görüntüdeki küçük lekeleri ve istenmeyen pikselleri temizler.

Yumuşak Geçiş: 
GaussianBlur ile oluşturulan alfa kanalı sayesinde, renk değişimi keskin hatlar yerine doğal bir geçişle uygulanır.

# Kullanılan Kütüphaneler ve Kurulum

OpenCV

Numpy

pip install opencv-python numpy

# Nasıl Çalışır?

1. Kırmızı Pikselleri Bul:
HSV renk uzayında kırmızının iki aralığı (0–10 ve 120–180) birleştirilerek gül pikselleri tespit edilir.

2. GrabCut — İki Aşamalı:
Önce dikdörtgen ipucuyla ilk tahmin yapılır. Ardından kırmızı pikseller "kesin ön plan", görüntü kenarları ve üst bulanık alan "kesin arka plan" olarak işaretlenerek ikinci geçiş daha hassas bir maske üretir.

3. Yeşil & Bulanık Alanları Temizle:
Maskeden yeşil yapraklar ve doygunluğu düşük (gri/bulanık) pikseller çıkarılır, ardından kırmızı pikseller tekrar eklenerek gül korunur.

4. Maske Temizleme:
Morfolojik OPEN ile küçük gürültüler, CLOSE ile maskede kalan delikler giderilir. Bağlantılı bileşen analizi ile sadece en büyük nesne (gül) seçilir.

5. Mora Boyama:
HSV kanalları ayrıştırılır. Gül piksellerinin Hue değeri 140 (mor) olarak güncellenir, doygunluk +30 artırılarak canlılık verilir.

6. Siyah Arka Planla Birleştir:
Alfa maske ile gül ön plana, sıfır matris siyah arka plana karşılık gelir. İkisi çarpılarak nihai görüntü elde edilir.

#Input

![rose](https://github.com/user-attachments/assets/a7e0e7b8-7b0d-4f21-b0fe-c274e8591343)

#Output

![mor_gul_siyah_bg](https://github.com/user-attachments/assets/e6bbfa60-8fe9-4840-99a8-9662d6ea010b)

