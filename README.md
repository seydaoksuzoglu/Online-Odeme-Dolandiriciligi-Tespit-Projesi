# Online Ödeme Dolandırıcılığı Tespit Projesi

## Giriş
İnternet üzerinden ödeme, günümüzde en popüler işlem yöntemlerinden biridir. Ancak, çevrimiçi ödemelerdeki artışla birlikte ödeme dolandırıcılııklarında da bir yükseliş gözlemlenmektedir. 
Bu proje, denetimli ve denetimsiz öğrenme tekniklerini kullanarak çevrimiçi ödeme işlemlerinde dolandırıcılığı tespit etmeyi amaçlamaktadır. Dolandırıcılığı doğru ve verimli bir şekilde tespit etmek için çeşitli makine öğrenmesi modelleri test edilmiş ve değerlendirilmiştir.

Veri seti 10 değişkenden oluşmaktadır:
* **step:** adım
* **type:** çevrimiçi işlem türü
* **amount:** işlem miktarı
* **nameOrig:** işlemi başlatan müşteri
* **oldbalanceOrg:** işlem öncesi bakiye
* **newbalanceOrig:** işlem sonrası bakiye
* **nameDest:** işlemin alıcısı
* **oldbalanceDest:** alıcının işlem öncesi başlangıç bakiyesi
* **newbalanceDest:** alıcının işlem sonrası yeni bakiyesi
* **isFraud:** dolandırıcılık işlemi

## Python Kütüphaneleri
pandas, numpy, matplotlib, scikit-learn

## Özellikler
- Dolandırıcılığı tespit etmek için denetimli ve denetimsiz öğrenme modelleri uygulanmıştır.
- Veri ön işleme, özellik seçimi ve model değerlndirme
- Değerlendirme metrikleri: Doğruluk (Precision), Duyarlılık (Recall), F1 Skoru

## Kullanılan Denetimli Öğrenme Modelleri
Proje esnasında birçok model denedim. En iyi performans gösteren ve doğruluk değeri en yüksek olan modeli kullanmayı amaçladım.
Bu modeller:
- Logistic Regression
- Decision Three Classifier
- Random Forest Classifier
- LinearSVC

Bu modeller içinde LinearSVC ve Logistic Regression her ne kadar iyi performans gösterse de en iyi doğruluğu veren model Random Forest Classifier oldu.
Bu sebepten bende denetimli öğrenme modelleri arasında bu modeli kullanmayı uygun buldum.
Şimdi modellerin verdiği doğruluk oranlarını inceleyelim.

**Logistic Regression:**

```
precision    recall  f1-score   support

           0       1.00      1.00      1.00   1906353
           1       0.79      0.58      0.67      2433

    accuracy                           1.00   1908786
   macro avg       0.90      0.79      0.83   1908786
weighted avg       1.00      1.00      1.00   1908786
```

* **Class 0 (Dolandırıcılık olmayan)** sınıfında model mükemmel performans gösteriyor. Precision, Recall ve F1-score epsi 1.00, yani model dolandırıcılık olmayan sınıfı çok iyi tahmin ediyor.
* **Class 1 (Dolandırıcılık olan)** sınıfında ise Precision %79, Recall %58 ve F1-score %67. Bu değerler, dolandırıcılık sınıfında bazı yanlış pozitif ve yanlış negatif tahminlerin yapıldığını gösteriyor.

* Veri setimizin büyük çoğunluğu dolandırıcılık olmayan sınıftan oluşuyor, bu nedenle model bu sınıfı çok iyi tahmin ederken, dolandırıcılık olan sınıfta performansı düşük kalıyor.
* Recall %58, yani dolandırıcılık olan örneklerin yaklaşık %42'si model tarafından kaçırılıyor. Bu, sahtecilik tespiti için önemli bir metrik.

**Decision Three Classifier:**

```
precision    recall  f1-score   support

           0       1.00      1.00      1.00      2982
           1       0.50      0.67      0.57        18

    accuracy                           0.99      3000
   macro avg       0.75      0.83      0.78      3000
weighted avg       0.99      0.99      0.99      3000
```

* **Class 0 (Dolandırıcılık olmayan)** sınıfındaki performans mükemmel, precision, recall ve f1-score değerleri 1'dir.
* **Class 1 (Dolandırıcılık olan)** sınıfında ise performans daha düşük. Recall değeri %67, bu da modelin bazı dolandırcılık vakalarını
kaçırdığını gösteriyor. Precision %50, yani dolandırıcılık olarak tahmin edilen vakaların yarısı yanlış pozitif.

* Veri setimizde dolandırcılık olmayan sınıf (0) çok baskın, bu yüzden model **class 0**'ı çok iyi tahmin ediyor. Ancak dolandırıcılık olan
sınıf(1) sayıca az olduğu için modelin performansı düşüyor.
* Ayrıca fark edebileceğiniz üzere veri setinde dolandırıcık ve dolandırıcılık olmayan çoğu veri alınmamış durumda. Bunun sebebi, modeli çalıştırdığım notebook'ta TPU kullanmama rağmen
çökmeler yaşanmasıdır. Bu sebepten `nrows=30000` yaptık. Yani, model kompleks olduğundan ve karmaşık dallanma içerdiğinden ağaç derinliği arttıkça, hesaplanması gereken düğüm sayısı ve düğümlerin oluşturulması için yapılan işlem miktarı da artar. Bu da çökmelere sebebiyet verebilir.
Sonuç olarak, Decision Three Classifier modeli iyi performans gösterememektedir.

**Random Forest Classifier:**

```
precision    recall  f1-score   support

           0       1.00      1.00      1.00      8979
           1       1.00      0.57      0.73        21

    accuracy                           1.00      9000
   macro avg       1.00      0.79      0.86      9000
weighted avg       1.00      1.00      1.00      9000
```

* **Class 0 (Dolandırıcılık olmayan)** sınıfı için mükemmel sonuçlar elde edilmiş: Precision, recall ve f1-score değerleri %100, yani bu sınıf tamamen doğru tahmin edilmiştir.
* **Class 1 (Dolandırıcılık olan)** sınıf için precision %100, ancak recall %57, bu da azınlık sınıfının doğru tahmin edilme oranının düşük olduğunu gösteriyor. Bu durum, modelin çoğunluk sınıfı üzerinde daha iyi performans gösterdiğini
ve azınlık sınıfını yeterince iyi tahmin etmediğini gösterir.
Accuracy %100 gibi görünse de, bu yüksek doğruluk oranı çoğunluk sınıfının baskın olmasından kaynaklanmaktadır ve azınlık sınıfının performansını yansıtmayabilir.
* Yine aynı şekilde veriler eksik durumda. Bunun sebebi model oldukça karmaşık olduğu için sürekli çökme yaşanmasıdır. Aynı şekilde `nrows=30000` yaptığımızda bu çökmelerin önüne geçebildik.
Doğruluk değerleri iyi görünse de model iyi performans gösteremediği için bu veri seti üzerinde kullanmadım.

**Linear SVC:**

```
precision    recall  f1-score   support

           0       1.00      1.00      1.00   1906257
           1       0.88      0.79      0.83      2529

    accuracy                           1.00   1908786
   macro avg       0.94      0.89      0.92   1908786
weighted avg       1.00      1.00      1.00   1908786
```

* **Class 0 (Dolandırıcılık olmayan)** sınıfı için, precision, recall ve F1-score %100, yani model bu sınıf için mükemmel şekilde tahmin ediyor.
* **Class 1 (Dolandırıcılık olan)** sınıfı için, precision %88, recall %79, ve F1-score %83, yani model bu sınıf içinde oldukça iyi tahminler yapıyor.
* **Genel Accuracy**, %100'dür. Bu da modelin tüm verilerdeki tahminlerin büyük çoğunluğunu doğru yaptığını gösterir.
* **Macro Average**, precision %94, recall %89 ve F1-score %92, yani her iki sınıfın performansını dikkate alarak genel başarım iyi düzeyde.
* **Weighted Average**, Precision, recall ve f1-score %100, çoğunluk sınıfının etkisi altında olan genel performansı yansıtır.

**Peki Neden Linear SCV'yi Seçmeliyiz?

## Kullanılan Denetimsiz Öğrenme Modelleri
- KMeans 

