# Skeleton-MixFormer [ACMMM 2023]
This repo is the official implementation for Skeleton-MixFormer: <u>Multivariate Topology Representation for Skeleton-based Action Recognition<u>


## Abstract
Vision Transformer, which performs well in various vision tasks, encounters a bottleneck in skeleton-based action recognition and falls short of advanced GCN-based methods. The root cause is that the current skeleton transformer depends on the self-attention mechanism of the complete channel of the global joint, ignoring the highly discriminative differential correlation within the channel, so it is challenging to learn the expression of the multivariate topology dynamically. To tackle this, we present Skeleton MixFormer, an innovative spatio-temporal architecture to effectively represent the physical correlations and temporal interactivity of the compact skeleton data.  Two essential components make up the proposed framework: 1) Spatial MixFormer. The channel-grouping and mix-attention are utilized to calculate the dynamic multivariate topological relationships. Compared with the full-channel self-attention method, Spatial MixFormer better highlights the channel groups' discriminative differences and the joint adjacency's interpretable learning. 2) Temporal MixFormer, which consists of Multiscale Convolution, Temporal Transformer and Sequential Holding Module. The multivariate temporal models ensure the richness of global difference expression and realize the discrimination of crucial intervals in the sequence, thereby enabling more effective learning of long and short-term dependencies in actions.
Our Skeleton MixFormer demonstrates state-of-the-art (SOTA) performance across seven different settings on four standard datasets, namely NTU-60, NTU-120, NW-UCLA, and UAV-Human.



# Dependencies

+ Python >= 3.6

+ PyTorch >= 1.1.0

+ PyYAML, tqdm, tensorboardX

# Data Preparation

### Download datasets

**There are 4 datasets to download:**
+ NTU RGB+D 60 Skeleton
+ NTU RGB+D 120 Skeleton
+ NW-UCLA
+ UAV-Human

**NTU RGB+D 60 and 120**

1. Request dataset: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:  
    i. ```nturgbd_skeletons_s001_to_s017.zip``` (NTU RGB+D 60)  
    ii. ```nturgbd_skeletons_s018_to_s032.zip``` (NTU RGB+D 120)  
    iii. Extract above files to ```./data/nturgbd_raw```  

**UAV-Human**

1. Download dataset from here: https://sutdcv.github.io/uav-human-web/
2. Move ```Skeleton``` to ```./data/UAV-Human```

**NW-UCLA**

1. Download dataset from [here](https://drive.google.com/file/d/1wWhgqMEQlrCKcJHu6W72Zk_iloS7_JJw/view?usp=share_link)
2. Move ```all_sqe``` to ```./data/NW-UCLA```



### NTU Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:
~~~
- data/
  - UAV-Human/
    - Skeleton
      ... # raw data of UAV-Human
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
~~~

#### Generating Data

+ Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:
~~~
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
~~~

### UAV Data Processing

#### Changes to statistics  
+ **Annotations**  

1. FileName: **P**000S00G10B10H10UC022000LC021000**A**000**R**0_08241716.txt  

2. **P**000: (**P**ersonID) unique person ID for the main subject in current video

3. **A**000: (**A**ction) action labels of current sample  

4. **R**0: (**R**eplicate) replicate capturing  

According to the organization form of UAV-human data set file name, change the person ID(**P**), the number of action repetition (**R**), action classification (**A**) and camera ID(**C**) in static data. Due to different collection methods of data sets, the default uav data is collected by a single camera, so the camera ids corresponding to all samples are set to 0.

#### Changes to code 

1. ```get_raw_skes_data.py``` Change the **ske_path** of the raw dataset, file extension, file name truncion method, and the size of the generated array used to store the coordinate information of the skeleton node in the current frame.

2. ```get_raw_denoisded_data.py``` set **noise_len_thres** = 0, Changing action label truncion way and all the numbers in the code from 25 to 17, 75 to 51, and 150 to 102. 

3. ```seq_transformation.py``` Classify the training and testing according to the https://github.com/SUTDCV/UAV-Human. 
 

#### Generate Data:

+ Generate UAV-Human dataset:
~~~
 cd ./data/uav/Skeleton
 # Updata statistics.py
 python updata_statistics.py
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
~~~
The pre-processed UAV-Human_CSv1 data can be referred [here](https://drive.google.com/file/d/1JsEWFlbfr8MM6U-7vbefww6jZqwna6ke/view?usp=sharing) and the pre-processed UAV-Human_CSv2 data can be referred [here](https://drive.google.com/file/d/1MwN4iNChfAza8cgJ_T2JCRm2P6rDm6ni/view?usp=sharing)
    
# Training & Testing
### Training
+ Change the config file depending on what you want.
~~~
    # Example: training SKMIXF on NTU RGB+D cross subject with GPU 0
    python main.py --config config/nturgbd-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/skmixf --device 0
      python main.py --config config/nturgbd-cross-subject/default.yaml --work-dir work_dir/ntu/csub/skmixf --device 0 1 2 3
    # Example: training provided baseline on NTU RGB+D cross subject
    python main.py --config config/nturgbd-cross-subject/default.yaml --model model.baseline.Model--work-dir work_dir/ntu/csub/baseline --     device 0
~~~
+ To train model on NTU RGB+D 60/120 with bone or motion modalities, setting ```bone``` or ```vel``` arguments in the config file ```default.yaml``` or in the command line.
~~~
    # Example: training SKMIXF on NTU RGB+D 120 cross subject under bone modality
    python main.py --config config/nturgbd120-cross-subject/default.yaml --train_feeder_args bone=True --test_feeder_args bone=True --work-     dir work_dir/ntu120/csub/skmixf_bone --device 0
~~~
+ To train model on NW-UCLA with bone or motion modalities, you need to modify ```data_path``` in ```train_feeder_args``` and ```test_feeder_args``` to "bone" or "motion" or "bone motion", and run
~~~
    python main.py --config config/ucla/default.yaml --work-dir work_dir/ucla/skmixf_xxx --device 0
~~~
+ To train model on UAV-Human with bone or motion modalities, you need to modify ```data_path``` in ```train_feeder_args``` and ```test_feeder_args``` to "bone" or "motion" or "bone motion", and run
~~~
    python main.py --config config/uav/default.yaml --work-dir work_dir/uav/skmixf_xxx --device 0
~~~

### Testing

+ To test the trained models saved in <work_dir>, run the following command:  

~~~
    python main.py --config /data/yinbaiqiao/Skeleton-MixFormer-main/config/nturgbd120-cross-set/default.yaml --work-dir /data/yinbaiqiao/Skeleton-MixFormer-main/work_dir/ntu120/cset --phase test --save-score True --weights /data/yinbaiqiao/Skeleton-MixFormer-main/NTU_120_Cset_J-8.pt  --device 4 5 6 7
~~~

+ To ensemble the results of different modalities, run  

~~~
    # Example: ensemble four modalities of SkMIXF on NTU RGB+D cross subject
    python ensemble.py --dataset ntu/xsub  --joint-dir  work_dir/ntu/csub/skmixf --bone-dir  work_dir/ntu/csub/skmixf_bone --joint-motion-dir  work_dir/ntu120/csub/skmixf_motion  --bone-motion-dir work_dir/ntu/csub/skmixf_bone_motion  --joint-k2-dir work_dir/ntu120/csub/skmixf_joint_k2  --joint-motion-k2-dir  work_dir/ntu120/csub/skmixf_joint_motion_k2
~~~

### Pretrained model
+ Pretrained weights for NTU RGB+D 60 and 120 can be downloaded from the following link [[Google Drive]](https://drive.google.com/file/d/15Ahneq5_IgurficrYb3PiiLeEFyS8lBQ/view?usp=share_link)
    
## Acknowledgements
This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) and [Info-GCN](https://github.com/stnoah1/infogcn) The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

Thanks to the original authors for their work!  


