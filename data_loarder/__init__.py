from torch.utils.data import DataLoader
from data_loarder.load_coco2017 import COCOSegmentation


def makeDataLoader(batchsize=16):
    works = 8
    trainData = COCOSegmentation(base_dir='/home/chengxiao/coco2017/', split='train', year=2017)
    trainLoader = DataLoader(trainData, num_workers=8, batch_size=batchsize, shuffle=True)
    valData = COCOSegmentation(base_dir='/home/chengxiao/coco2017/', split='val', year=2017)
    valLoader = DataLoader(valData, num_workers=8, batch_size=batchsize, shuffle=True)
    return trainLoader, valLoader
