from PIL import Image
import numpy as np

import torch
import torchvision.transforms as tr

from model.deeplab import DeepLab
from customer_utils.lable_img import decode_segmap

device = torch.device('cuda:0')

checkpoint = torch.load("/home/chengxiao/pycharmProject/myDeepLabV3Plus/ckp/resnet101/model79_0.6293811152760053.pth")

model = DeepLab(num_classes=21,
                backbone='mobilenet',
                output_stride=16,
               )
# model = torch.nn.DataParallel(model)

model.load_state_dict(checkpoint)
model.eval()
model.to(device)

def transform(image):
    return tr.Compose([
        # tr.Resize(513),
        # tr.CenterCrop(513),
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])(image)

torch.set_grad_enabled(False)

image = Image.open('person_with_car1.jpg')
inputs = transform(image).to(device)
print("one")
output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
print("one")
pred = np.argmax(output, axis=0)

# Then visualize it:
decode_segmap(pred, dataset="coco", plot=True)