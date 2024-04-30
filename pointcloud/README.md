# Applying Context CLuster to Point Cloud Analysis

Our point cloud classification  implementation is based on [pointMLP](https://github.com/ma-xu/pointMLP-pytorch). Thank the authors for their wonderful works.

## Note
Please note that we just simply follow the hyper-parameters of pointMLP which may not be the optimal ones for Context Cluster. 
Feel free to tune the hyper-parameters to get better performance. 


## Usage

Install pointMLP required libs, see [README in pointMLP](https://github.com/ma-xu/pointMLP-pytorch). 


## Data preparation

We don't need to download the ScanObjectNN dataset by ourself. The dataset will be automatically downloaded at the first running. 


## Results and models

| Method | mACC | OA | Download |
| --- | --- | --- | --- |
| PointMLP_CoC | 84.4| 86.2| [log & model](https://drive.google.com/drive/folders/1R5nQTp9mnza3FdqA0FosRj_mzvn4He1F?usp=sharing) |


## Train
To train pointMLP_CoC, run:
```
python main.py --model pointMLP --model2 Model
python main.py --model PointConT
python main.py --model MixModel
```
python main.py --config /home/huangjiehui/Project/HDAnaylis/ybq/video-analysis/Context-Cluster-main/pointcloud/config/default.yaml --model pointMLP_CoC 
python main.py --feeder feeders.feeder_ntu.Feeder --train-feeder-args  {data_path: /home/huangjiehui/Project/HDAnaylis/ybq/video-analysis/CTR-GCN-main/data/ntu/NTU60+_CS.npz split: train }--test_feeder_args  {data_path: /home/huangjiehui/Project/HDAnaylis/ybq/video-analysis/CTR-GCN-main/data/ntu/NTU60+_CS.npz split: test --model pointMLP_CoC}

ma-xu said:
x is feature map and xyz is the original coordinates.

k-nearest is based on coordinates, that's why we need xyz.

Do we need xyz at all if use_xyz = False? We still need it for LocalGrouper. use_xyz means if we will add xyz to x.
