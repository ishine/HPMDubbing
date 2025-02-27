# HPMDubbing🎬 - PyTorch Implementation

In this [paper](), we propose a novel movie dubbing architecture via hierarchical prosody modeling, which bridges the visual information to corresponding speech prosody from three aspects: lip, face, and scene. Specifically, we align lip movement to the speech duration, and convey facial expression to speech energy and pitch via attention mechanism based on valence and arousal representations inspired by the psychology findings. Moreover, we design an emotion booster to capture the atmosphere from global video scenes. All these embeddings are used together to generate mel-spectrogram, which is then converted into speech waves by an existing vocoder. Extensive experimental results on the V2C and Chem benchmark datasets demonstrate the favourable performance of the proposed method.


[//]: # (We provide our implementation and pre-trained models as open-source in this repository. )

[//]: # (&#40;Continue to upload, before the upload is finished, don't rush to run 🌟&#41;)

____________________________
🌟🌟🌟🥳 Here is a display of the Demo generated by HPMDubbing:

https://user-images.githubusercontent.com/109259667/230555695-d0a9f2bd-82f9-4448-b9be-3424b217cbfc.mp4

📝Text: That defines our equilibrium.


https://user-images.githubusercontent.com/109259667/230560457-5656856d-1a08-424d-9d2b-746a2615d1cf.mp4

📝Text: Each gas will exert what's called a partial pressure.


https://user-images.githubusercontent.com/109259667/230557290-c3fd3ed5-112e-49fa-bc5b-b5001caf6644.mp4

📝Text: Yes. I'm the baby Jesus.


https://user-images.githubusercontent.com/109259667/230558797-b08aaf0a-f27c-4a6d-9d95-a550c7145acf.mp4

📝Text: This? No! Oh, no. This is just a temporary thing.





https://user-images.githubusercontent.com/109259667/232961804-d8f43c7e-869d-4a50-871e-69de29946d82.mp4


📝Text: It was an accident. She was scared.

Visit our [demo website]() or download the generated samples by HPMDubbing model (🔊[result on Chem](https://drive.google.com/drive/folders/1C3YUngeH0ENqr9erRM2iOFnjr7VN9tPc?usp=share_link) and 🔊[result on V2C](https://drive.google.com/drive/folders/1-lbx9xH0hTpV4ngdyxJOrvtOyPMVBG_e?usp=share_link)) to see more results.

# Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

# Dataset
## 1) For V2C
[V2C-MovieAnimation](https://github.com/chenqi008/V2C) is a multi-speaker dataset for animation movie dubbing with identity and emotion annotations. It is collected from 26 Disney cartoon movies and covers 153 diverse characters. Due to the copyright, we can not directly provide the dataset, [see V2C issue](https://github.com/chenqi008/V2C/issues/1).

In this work, we release the [V2C-MovieAnimation2.0](https://pan.baidu.com/s/151ljJuY72bkntxxgEIUTkQ) to satisfy the requirement of dubbing the specified characters. 
Specifically, we removed redundant character faces in movie frames (please note that our video frames are sampled at 25 FPS by ffmpeg). 
You can download our preprocessed features directly through the link 
[GoogleDrive](https://drive.google.com/drive/folders/1AB-E682_OYhFSBz-y1t36A45e9l1ssRd?usp=share_link) or [BaiduDrive](https://pan.baidu.com/s/1ucVl116YcEBvlZopd9W5mw) (password: Good).
![Illustration](./images/Our_V2C2.0_Illustration.jpeg)

## 2) For Chem
The Chem dataset is provided by [Neural Dubber](https://tsinghua-mars-lab.github.io/NeuralDubber/), which belongs to the single-speaker chemistry lecture dataset from [Lip2Wav](https://github.com/Rudrabha/Lip2Wav).
<!-- 3) For the chem dataset, we are very grateful to Ph.D. Hu and his team for providing us with this dataset. They segment the long videos into sentence-level clips according to the start and end timestamp of each sentence in the transcripts. If you need a citation, [please click their project link](https://tsinghua-mars-lab.github.io/NeuralDubber/), thanks! -->

# Data Preparation

For voice preprocessing (mel-spectrograms, pitch, and energy), Montreal Forced Aligner (MFA) is used to obtain the alignments between the utterances and the phoneme sequences. Alternatively, you can skip the below-complicated step, and use our extracted features, directly.

Download the official [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) package and run
```
./montreal-forced-aligner/bin/mfa_align /data/conggaoxiang/HPMDubbing/V2C_Data/wav16 /data/conggaoxiang/HPMDubbing/lexicon/librispeech-lexicon.txt  english /data/conggaoxiang/HPMDubbing/V2C_Code/example_V2C16/TextGrid -j
```
then, please run the below script to save the .npy files of mel-spectrograms, pitch, and energy from two datasets, respectively.
```
python V2C_preprocess.py config/MovieAnimation/preprocess.yaml
```
```
python Chem_preprocess.py config/MovieAnimation/preprocess.yaml
```
For hierarchical visual feature preprocessing (lip, face, and scenes), we detect and crop the face from the video frames using $S^3FD$ [face detection model](https://github.com/yxlijun/S3FD.pytorch). Then, we align faces to generate 68 landmarks and bounding boxes (./landmarks and ./boxes). Finally, we get the mouth ROIs from all video clips, following [EyeLipCropper](https://github.com/zhliuworks/EyeLipCropper). Similarly, you can also skip the complex steps below and directly use the features we extracted.

We use the pre-trained weights of [emonet](https://github.com/face-analysis/emonet) to extract affective display features, and fine-tune Arousal and Valence (dimension256) according to the last layer of emonet network.
```
python V2C_emotion.py -c emonet_8.pth -o /data/conggaoxiang/V2C_feature/example_V2C_framelevel/MovieAnimation/VA_feature -i /data/conggaoxiang/detect_face 
```
The lip feature is extracted by [resnet18_mstcn_video](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks), which inputs the grayscale mouth ROIs for each video.
```
python lip_main.py --modality video --config-path /data/conggaoxiang/lip/Lipreading_using_Temporal_Convolutional_Networks-master/configs/lrw_resnet18_mstcn.json --model-path /data/conggaoxiang/lip/Lipreading_using_Temporal_Convolutional_Networks-master/models/lrw_resnet18_mstcn_video.pth --data-dir /data/conggaoxiang/lip/Lipreading_using_Temporal_Convolutional_Networks-master/MOUTH_processing --annonation-direc /data/conggaoxiang/lip/LRW_dataset/lipread_mp4 --test
```
Finally, the scenes feature is provided by V2C-Net from [I3D model](https://github.com/piergiaj/pytorch-i3d).
```
python ./emotion_encoder/video_features/emotion_encoder.py
```

# Vocoder
We provide the pre-trained model and implementation details of [HPMDubbing_Vocoder](https://github.com/GalaxyCong/HPMDubbing_Vocoder). Please download the vocoder of HPMDubbing and put it into the `vocoder/HiFi_GAN_16/` or `/vocoder/HiFi_GAN_220/` folder.
Before running, remember to check line 63 of `model.yaml` and change it to your own path. 
```
vocoder:
  model: [HiFi_GAN_16] or [HiFi_GAN_220]
  speaker: "LJSpeech" 
  vocoder_checkpoint_path: [Your path]
```

# Training

For V2C-MovieAnimation dataset, please run train.py file with
```
python train.py -p config/MovieAnimation/preprocess.yaml -m config/MovieAnimation/model.yaml -t config/MovieAnimation/train.yaml -p2 config/MovieAnimation/preprocess.yaml
```
For Chem dataset, please run train.py file with
```
python train.py -p config/Chem/preprocess.yaml -m config/Chem/model.yaml -t config/Chem/train.yaml -p2 config/Chem/preprocess.yaml
```
![Illustration](./images/train.jpeg)
# Pretrained models
You can also use pretrained models we provide, download pretrained models through the link 
[GoogleDrive](https://drive.google.com/drive/folders/1w3mAHsgN2MK20C3Mo6T_EWYdYifN3r8v?usp=share_link) or [BaiduDrive](https://pan.baidu.com/s/1yZCh57aJh-Dk6Mwai7eNgQ) (password: star). And synthesize the speech generated by the model through the following command:

```
python Synthesis.py --restore_step [Chekpoint] -p config/MovieAnimation/preprocess.yaml -m config/MovieAnimation/model.yaml -t config/MovieAnimation/train.yaml -p2 config/MovieAnimation/preprocess.yaml
```


# Tensorboard
Use
```
tensorboard --logdir output/log/MovieAnimation --port= [Your port]
```
or 
```
tensorboard --logdir output/log/Chem --port= [Your port]
```
to serve TensorBoard on your localhost.
The loss curves, mcd curves, synthesized mel-spectrograms, and audios are shown.


# References
- [V2C: Visual Voice Cloning](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_V2C_Visual_Voice_Cloning_CVPR_2022_paper.pdf), Q. Chen, *et al*.
- [Neural Dubber: Dubbing for Videos According to Scripts](https://proceedings.neurips.cc/paper/2021/file/8a9c8ac001d3ef9e4ce39b1177295e03-Paper.pdf), C. Hu, *et al*.
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.

# Citation
<!-- If you find our work useful in your research, please consider citing: -->
If our research and this repository are helpful to your work, please cite with:
```
@inproceedings{cong2023learning,
  title={Learning to Dub Movies via Hierarchical Prosody Models},
  author={Cong, Gaoxiang and Li, Liang and Qi, Yuankai and Zha, Zheng-Jun and Wu, Qi and Wang, Wenyu and Jiang, Bin and Yang, Ming-Hsuan and Huang, Qingming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14687--14697},
  year={2023}
}
```
