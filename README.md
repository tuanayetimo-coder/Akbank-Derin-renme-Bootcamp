# Akbank Derin Öğrenme Bootcamp Projesi

Bu proje, Akbank Derin Öğrenme Bootcamp kapsamında, TensorFlow ve Keras kütüphaneleri kullanılarak geliştirilen bir **Evrişimli Sinir Ağı (CNN)** modelini içermektedir.  
Projenin temel amacı, verilen bir görüntünün yüksek doğrulukla **kedi mi yoksa köpek mi** olduğunu sınıflandırmaktır.

---

##  Projenin Amacı ve Hedefleri

- Kedi ve köpek resimlerinden oluşan geniş bir veri seti üzerinde bir CNN modeli eğitmek.  
- Veri artırma (data augmentation) tekniklerini kullanarak modelin ezberlemesini (overfitting) önlemek ve genelleme yeteneğini artırmak.  
- Eğitilen modelin performansını doğruluk (accuracy) ve kayıp (loss) metrikleri üzerinden değerlendirmek.

---

##  Veri Kümesi

Popüler Kaggle "Cat and Dog" veri kümesi kullanılmıştır.

| Dizinin Adı         | Açıklama                                         |
|--------------------|-------------------------------------------------|
| Eğitim Verisi       | Modelin öğrenmesi için kullanılan binlerce kedi ve köpek görüntüsü. |
| Test (Doğrulama) Verisi | Modelin eğitimde görmediği ve performansını ölçmek için kullanılan görüntüler. |

---

## 🛠️ Uygulanan Adımlar ve Teknik Detaylar

### 1. Veri Yükleme ve Ön İşleme
- `ImageDataGenerator` sınıfı kullanılarak görüntüler dizinlerden okunmuştur.  
- Tüm görüntüler, modelin giriş katmanına uygun olacak şekilde **128x128** piksel boyutuna yeniden ölçeklendirilmiştir.  
- Piksel değerleri **0-1 aralığına** normalleştirilmiştir.

### 2. Veri Artırma (Data Augmentation)
- Modelin genelleme yeteneğini artırmak ve overfitting’i önlemek için eğitim verilerine rastgele dönüşümler uygulanmıştır:
  - `rotation_range`: Görüntüleri rastgele döndürme  
  - `width_shift_range` / `height_shift_range`: Yatay ve dikey kaydırma  
  - `shear_range`: Makaslama (kırpma)  
  - `zoom_range`: Rastgele yakınlaştırma  
  - `horizontal_flip`: Görüntüleri yatay çevirme

### 3. Model Mimarisi (CNN)
| Katman Tipi | Parametreler | Aktivasyon | Amaç |
|-------------|-------------|------------|------|
| Conv2D      | Çeşitli filtre sayıları | relu | Düşük ve orta seviyeli özellikleri çıkarma |
| MaxPooling2D | (2,2) | N/A | Boyut indirgeme ve önemli özellikleri koruma |
| Flatten     | N/A | N/A | 2D haritaları 1D vektöre dönüştürme |
| Dense (Gizli) | Çeşitli nöron sayıları | relu | Öğrenilen özellikleri harmanlama |
| Dense (Çıkış) | 1 nöron | sigmoid | İkili sınıflandırma çıktısı (0-1) |

### 4. Model Derleme ve Eğitimi
| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Optimizasyon | adam | Hızlı ve etkin optimizasyon |
| Kayıp Fonksiyonu | binary_crossentropy | İkili sınıflandırma problemleri için standart kayıp |
| Metrik | accuracy | Performans ölçümü |
| Epoch | 10 | Eğitim veri setinin kaç kez görüleceği |
| Eğitim Fonksiyonu | fit_generator | Veri artırma ile eğitim |

---

##  Sonuçlar ve Değerlendirme
- **Doğruluk:** Eğitim sonunda model, doğrulama verileri üzerinde yaklaşık **%80 doğruluk** sağlamıştır.  
- **Grafiksel Analiz:** Eğitim/doğrulama doğruluk ve kayıp grafikleri, modelin sağlıklı bir öğrenme süreci geçirdiğini ve overfitting’in büyük ölçüde önlendiğini göstermektedir.  
- **Çıkarım:** Geliştirilen CNN modeli, kedi ve köpek görüntülerini başarılı bir şekilde sınıflandırmaktadır.

---

##  Gelecek İyileştirmeler
- Epoch sayısını artırmak  
- Daha derin mimari kullanmak  
- Transfer Learning (VGG16, ResNet) ile performansı artırmak  

---

##  Proje Linki 

(https://www.kaggle.com/code/azraaltundasss/akbank-derin-renme-bootcamp-projesi) --> Projeyi kendi bilgisayarımın tamirde olması dolayısıyla arkadaşımın hesabından tasarlamaya başladım ancak son halini kendi bilgisayarımdan tamamladım. Bu yüzden ortak notebookta yer alıyor.

##  Gereksinimler
Python 3.x ve aşağıdaki kütüphaneler gereklidir:  

```txt
tensorflow
keras
numpy
matplotlib
pandas
opencv-python


