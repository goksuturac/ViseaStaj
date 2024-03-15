import torch
import pandas as pd

x = torch.rand(2,3)
# print(f"oluşturulan değer: {x} \noluşturulan değerin türü: {x.dtype} \n oluşturulan değerin boyutu: {x.size()} \n --------------------------------")

# unsqueezed_x = x.unsqueeze(0)
# print(f"unsqueezed(0) değer: {unsqueezed_x} \noluşturulan değerin türü: {unsqueezed_x.dtype} \nunsqueezed(0) değerin boyutu: {unsqueezed_x.size()} \n --------------------------------")

# unsqueezedd_x = x.unsqueeze(1)
# print(f"unsqueezed(1) değer: {unsqueezedd_x} \nunsqueezed(1) değerin türü: {unsqueezedd_x.dtype} \nunsqueezed(1) değerin boyutu: {unsqueezedd_x.size()} \n --------------------------------")

# unsqueezeddd_x = x.unsqueeze(2)
# print(f"unsqueezed(2) değer: {unsqueezeddd_x} \nunsqueezed(2) değerin türü: {unsqueezeddd_x.dtype} \nunsqueezed(2) değerin boyutu: {unsqueezeddd_x.size()}")


unsqueezed_x = x.unsqueeze(0)
unsqueezedd_x = x.unsqueeze(1)
unsqueezeddd_x = x.unsqueeze(2)

data_x = {
    "Yapılan İşlem|": ["torch.rand(2,3)","unsqueeze(0)","unsqueeze(1)","unsqueeze(2)"],
    "Değerin Türü|": [x.dtype, unsqueezed_x.dtype, unsqueezedd_x.dtype, unsqueezeddd_x.dtype],
    "Boyut|": [x.size(), unsqueezed_x.size(), unsqueezedd_x.size(), unsqueezeddd_x.size()],
    "Oluşturulan Değer": [x, unsqueezed_x, unsqueezedd_x, unsqueezeddd_x]
}

df = pd.DataFrame(data_x)
pd.set_option("display.max_colwidth", None)
print(df)

print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

y = torch.rand(4,5)
unsqueezed_y = y.unsqueeze(0)
unsqueezedd_y = y.unsqueeze(1)
unsqueezeddd_y = y.unsqueeze(2)

data_y = {
    "Yapılan İşlem|": ["torch.rand(4,5)","unsqueeze(0)","unsqueeze(1)","unsqueeze(2)"],
    "Değerin Türü|": [y.dtype, unsqueezed_y.dtype, unsqueezedd_y.dtype, unsqueezeddd_y.dtype],
    "Boyut|": [y.size(), unsqueezed_y.size(), unsqueezedd_y.size(), unsqueezeddd_y.size()],
    # "Oluşturulan Değer": [y, unsqueezed_y, unsqueezedd_y, unsqueezeddd_y]
}

df = pd.DataFrame(data_y)
# pd.set_option("display.max_colwidth", None)
print(df)
