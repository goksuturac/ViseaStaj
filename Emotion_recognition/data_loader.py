import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import os

class DataLoader:
    def __init__(self, data_dir, target_size=(224, 224),  random_state=None):
        self.data_dir = data_dir
        self.target_size = target_size
        self.random_state = random_state

    def load_data(self):
        classes = os.listdir(self.data_dir)
        class_indices = {cls_name: i for i, cls_name in enumerate(classes)}

        X = []
        y = []

        for cls_name in classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            for img_file in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.target_size)
                X.append(img)
                y.append(class_indices[cls_name])
        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        return X_train, X_test, y_train, y_test
    
    def __getitem__(self, index):
        X_train, X_test, y_train, y_test = self.load_data()
        if index < len(X_train):
            return X_train[index], y_train[index]
        else:
            index -= len(X_train)
            return X_test[index], y_test[index]
        
    def on_epoch_end(self):
        # Her bir epoch sonunda yapılacak işlemleri buraya yazılır
        pass

    def get_train_data_loader(self):
        # Eğitim veri yükleyiciyi oluştur
        #_ işareti load data fonksiyonunun döndürdüğü o değerlere ihtiyaç olmadığında kullanılmamak üzere yazıldı.
        X_train, _, y_train, _ = self.load_data()
        return X_train, y_train

    def get_test_data_loader(self):
        # Test veri yükleyiciyi oluştur
        _, X_test, _, y_test = self.load_data()
        return X_test, y_test