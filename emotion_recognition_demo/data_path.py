import glob
import random
import pandas as pd

train_datapath = "C:\\Users\ASUS\\Desktop\\Staj\\ViseaStaj\\emotion_recognition_demo\\dataset\\train"
test_datapath = "C:\\Users\ASUS\\Desktop\\Staj\\ViseaStaj\\emotion_recognition_demo\\dataset\\test"

train_path = []
classes = []

for data_path in glob.glob(train_datapath + '\\*'):
    classes.append(data_path.split('\\')[-1])
    train_path.extend(glob.glob(data_path + '\\*'))

train_image_paths = list(pd.core.common.flatten(train_path))
random.shuffle(train_image_paths)

test_image_paths = []
for data_path in glob.glob(test_datapath + '\\*'):
    test_image_paths.extend(glob.glob(data_path + '\\*'))

test_image_paths = list(pd.core.common.flatten(test_image_paths))
train_image_pathtest_image_pathss = train_image_paths[:int(0.8*len(train_image_paths))]

with open("C:\\Users\ASUS\\Desktop\\Staj\\ViseaStaj\\emotion_recognition_demo\\dataset\\train.txt", 'w') as f:
    for i in range(len(train_image_paths)):
        f.write(str(train_image_paths[i]))
        f.write('\n')

with open("C:\\Users\ASUS\\Desktop\\Staj\\ViseaStaj\\emotion_recognition_demo\\dataset\\test.txt", 'w') as f:
    for i in range(len(test_image_paths)):
        f.write(str(test_image_paths[i]))
        f.write('\n')
