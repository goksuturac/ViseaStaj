import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    #__init__ fonksiyonu --> başlatıcı fonksiyondur. DataŞLoader sınıfı çağrıldığında alacak olan parametreleri belirler. 
    def __init__(self, data_dir, target_size=(224,224), test_size=0.2, random_state=None):
        self.data_dir = data_dir
        self.target_size = target_size
        self.test_size = test_size
        self.random_state = random_state
        
    def load_data(self):
        #klasör içindeki dosyaların sınıflarının listesini alır.
        classes = os.listdir(self.data_dir)
        class_indices = {cls_name : i for i, cls_name in enumerate(classes)}
        
        #verisetini X ve y olarak ayırıyoruz.
        X = []
        y = []
        
        for cls_name in classes:
            cls_dir = os.path.join(self.data_dir,cls_name)
            for img_file in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.target_size)
            # X burda klasör içindeki fotoğrafları aldı
            X.append(img)
            # y burda o fotoğrafların ait olduğu sınıfın ismini aldı
            y.append(class_indices[cls_name])
        # eklenen fotoğraf ve sınıfları numpy array e çevirdik.
        X = np.array(X)
        y = np.array(y)
        
        # data ve label olarak X ve y ye ayırdığımız verisetimizi __init__ sınıfında tanımladığımız test_size kadar sklearn kütüphanesindeki train_test_split methıdunu kullanarak ayırdık.
        # train datasetimizdeki fotoğraf değerleri X_train ||  train datasetimizde fotoğrafların labelları y_train  ||  test datasetimizdeki fotoğraflar X_test  ||  test datasetimizdeki fotoğraflara karşılık gelen labellar y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test