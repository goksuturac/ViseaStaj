import glob
import os

'''
glob.glob(**/*.txt)--> sadece kodun çalıştığı yerdeki ilk txt dosyasını veriyor.
glob.glob(**/*.txt, root_dir="C:\\Users\\ASUS\\Desktop\\Staj\\ViseaStaj\\", recursive= False)--> o yolun içindeki ilk txt dosyasını veriyor.
glob.glob(**/*.txt, root_dir= "C:\\Users\\ASUS\\Desktop\\Staj\\ViseaStaj\\", recursive= True)--> o yolda bulunan tüm txt dosyalarını veriyor.
glob.glob(**/*.txt, root_dir= "C:\\Users\\ASUS\\Desktop\\Staj\\ViseaStaj\\", recursive= True, include_hidden = True)--> o yolda bulunan tüm gizli ve gizli olmayan txt dosyalarını veriyor.
'''

# path = "C:\\Users\\ASUS\\Desktop\\Staj\\ViseaStaj\\emotion_recognition_demo"

print(glob.glob("**/*.txt", recursive= True))

print(glob.glob('*.py'))

# print(glob.glob('emotion_recognition_demo/*.gitignore'))
# print(glob.glob(pathname=path), "*.txt")





# txt_files = glob.glob(path + "/**/*.txt", recursive=True)
# txt_files = [os.path.basename(file) for file in txt_files]

# print(glob.glob(path +"/**/[d].py"))
# print(txt_files)
