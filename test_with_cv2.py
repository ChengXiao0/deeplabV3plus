import cv2
from PIL import Image
import numpy as np

import torch
# from torchvision import transforms as tr
import torchvision.transforms as tr

from model.deeplab import DeepLab
from customer_utils.lable_img import decode_segmap

device = torch.device('cuda:0')

checkpoint = torch.load("ckp/model59_0.42959582026409104.pth")

model = DeepLab(num_classes=21,
                backbone='resnet',
                output_stride=16,
                )
model = torch.nn.DataParallel(model)
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
video = cv2.VideoCapture("./testVideo.mp4")
print(video.get(3))
print(video.get(4))
print(video.get(5))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
wv = cv2.VideoWriter('output.avi', fourcc, 15.0, (368, 272))
while (video.isOpened()):
    ret, frame = video.read()
    if not ret:
        wv.release()
        break
    imgFromVideo = Image.fromarray(frame.copy())
    inputs = transform(imgFromVideo).to(device)
    output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
    pred = np.argmax(output, axis=0)
    out = decode_segmap(pred, dataset="coco", plot=False)
    wv.write(out)
    cv2.namedWindow("out")
    cv2.namedWindow("frame")
    cv2.imshow('out', out)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
# image = Image.open('IMG_20210510_162630.jpg')
# inputs = transform(image).to(device)
# output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
# pred = np.argmax(output, axis=0)
#
# # Then visualize it:
# decode_segmap(pred, dataset="coco", plot=True)
