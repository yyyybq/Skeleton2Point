import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))



def load_scanobjectnn_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")#读取h5f文件
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_NTU_data(partition):
    Feeder = import_class(self.arg.feeder)
    data_loader = dict()
    if partition == 'train':
            data_loader['train'] =Feeder(**self.arg.train_feeder_args)         
    else:
            data_loader['test'] =Feeder(**self.arg.test_feeder_args)                

    return all_data

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])#类似于randint，但是返回float,均匀分布随机采样
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')#内积
    return translated_pointcloud


class NTU(Dataset):

    def __init__(self, num_points, partition='training'):
        self.data= load_NTU_data(partition)
        self.label=np.loadtxt('/home/huangjiehui/Project/HDAnaylis/ybq/video-analysis/CTR-GCN-main/data/ntu/statistics/label.txt', dtype=np.int) - 1 
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    n=300*25
    train = NTU(n)
    test = NTU(n, 'test')
    for data, label in train:
        print(data.shape)
        print(label)