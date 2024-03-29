import glob
import random
from pandas.core.common import flatten as pdf

train_datapath = "emotion_recognition_demo\\dataset\\train"
test_datapath = "emotion_recognition_demo\\dataset\\test"

train_path = []
classes = []

for data_path in glob.glob(train_datapath + '\\*'):
     classes.append(data_path.split('\\')[-1])
     train_path.extend(glob.glob(data_path + '\\*'))
     

train_image_paths = list(pdf(train_path))
random.shuffle(train_image_paths)

test_image_paths = []
for data_path in glob.glob(test_datapath + '\\*'):
    test_image_paths.extend(glob.glob(data_path + '\\*'))

test_image_paths = list(pdf(test_image_paths))

with open("emotion_recognition_demo\\dataset\\train.txt", 'w') as x:
    for i in range(len(train_image_paths)):
        x.write(str(train_image_paths[i]))
        x.write('\n')

with open("emotion_recognition_demo\\dataset\\test.txt", 'w') as x:
    for i in range(len(test_image_paths)):
        x.write(str(test_image_paths[i]))
        x.write('\n')
