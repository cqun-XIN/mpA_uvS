# mpA uvS: Multi-Perspective Attention for Unsupervised Video Summarization: Capturing Global, Local, and Spatio-temporal Context
mpA uvS is an unsupervised video summarization model that leverages multi-perspective attention to evaluate the importance of video frames. The model integrates information from global, local, and spatio-temporal contexts to capture contextual semantic dependencies between video frames. Global attention captures dependencies across the entire video, while local attention focuses on segments within the video. Spatio-temporal attention examines frames at the same spatial positions to identify temporal changes in important objects. By incorporating uniqueness and diversity metrics, mpA uvS generates summaries that are representative and non-redundant. Experimental results on the SumMe and TVSum datasets demonstrate that mpA uvS outperforms state-of-the-art unsupervised models and remains competitive with supervised approaches, while achieving efficient training times.
# Requirements
Python 3.8(.8)\
PyTorch 1.7.1\
CUDA Version 11.0\
cuDNN Version 8005\
TensorBoard 2.4.1\
TensorFlow 2.3.0\
NumPy 1.20.2\
H5py 2.10.0
# Data
Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the [data]() folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao](https://github.com/pujols) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). These files have the following structure:\

  

| /key |  |  
| ------------- | ------------- |  
| /features  | 2D-array with shape (n_steps, feature-dimension)  |  
| /gtscore  | 1D-array with shape (n_steps), stores ground truth importance score (used for training, e.g. regression loss)  |  
|/user_summary | 2D-array with shape (num_users, n_frames), each row is a binary vector (used for test) |
|/change_points  |         2D-array with shape (num_segments, 2), each row stores indices of a segment |  
|/n_frame_per_seg  |       1D-array with shape (num_segments), indicates number of frames in each segment |  
   | /n_frames     |            number of frames in original video |  
   | /picks      |              positions of subsampled frames in original video |  
   | /n_steps     |             number of subsampled frames |  
   | /gtsummary   |             1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood) |  
   | /video_name (optional)  |  original video name, only available for SumMe dataset  
   
   Original videos and annotations for each dataset are also available in the dataset providers' webpages:
   [SumMe](https://gyglim.github.io/me/vsum/index.html#benchmark) [TVSum](https://github.com/yalesong/tvsum)
   # Training  
   To train the model using one of the aforementioned datasets and for a number of randomly created splits of the dataset (where in each split 80% of the data is used for training and 20% for testing) use the corresponding JSON file that is included in the data/splits directory. This file contains the 5 randomly-generated splits that were utilized in our experiments.  
   For training the model using a single split, run:  
  ```
  python model/main.py --split_index N --n_epochs E --batch_size B --video_type 'dataset_name'  
  ```
where, `N` refers to the index of the used data split, `E` refers to the number of training epochs, `B` refers to the batch size, and `dataset_name` refers to the name of the used dataset.  

Alternatively, to train the model for all 5 splits, use the `run_summe_splits.sh` and/or `run_tvsum_splits.sh` script and do the following:  
```
chmod +x model/run_summe_splits.sh    # Makes the script executable.
chmod +x model/run_tvsum_splits.sh    # Makes the script executable.
./model/run_summe_splits.sh           # Runs the script. 
./model/run_tvsum_splits.sh           # Runs the script.
```
Please note that after each training epoch the algorithm performs an evaluation step, using the trained model to compute the importance scores for the frames of each video of the test set. These scores are then used by the provided evaluation scripts to assess the overall performance of the model (in F-Score).
# Configurations  
The utilized model selection criterion relies on the post-processing of the calculated losses over the training epochs and enables the selection of a well-trained model by indicating the training epoch. To evaluate the trained models of the architecture and automatically select a well-trained model, define the `dataset_path` in `compute_fscores.py` and run `evaluate_exp.sh`. To run this file, specify:  
- `base_path/exp$exp_num:` the path to the folder where the analysis results are stored,
- `$dataset:` the dataset being used, and
- `$eval_method:` the used approach for computing the overall F-Score after comparing the generated summary with all the available user summaries (i.e., 'max' for SumMe and 'avg' for TVSum).
```
  sh evaluation/evaluate_exp.sh $exp_num $dataset $eval_method
```
# Acknowledgement  
We referenced the repos below for the code  
[CA-SUM](https://github.com/e-apostolidis/CA-SUM)
