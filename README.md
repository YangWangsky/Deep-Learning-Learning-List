# Deep-Learning-Learning-List

Some courses or papers I have learned

## Table of Contents

- [Online Courses](#online-courses)
- [Papers](#papers)
  - [Survey](#survwy)
  - [Image Classification](#image-classification)
    - [Backbone](#backbone)
    - [Visualization](#visualization)
    - [Cls Others](#cls-others)
  - [Video](#video)
  - [Object Detection](#object-detection)
    - [One-Stage](#one-stage)
    - [Two-Stage](#two-stage)
  - [Segmentation](#segmentation)
    - [Semantic Segmentation](#semantic-segmentation)
    - [Real time Semantic Segmentation](#real-time-semantic-segmentation)
    - [Point Cloud](#point-cloud)
    - [Instance segmentation](#instance-segmentation)
    - [Panoptic segmentation](#panoptic-segmentation)
  - [Domain Adaptation](#domain-adaptation)
  - [Nautre Language Process](#nautre-language-process)
  - [EEG](#eeg)
    - [Generic](#generic-eeg)
    - [Sleep Stage Classification](#sleep-stage-classification)
- [Software and Skills](#software-and-skills)

## Online Courses

- [Andrew Ng. Machine learning. Stanford.](https://www.coursera.org/learn/machine-learning/home/welcome)
- [Andrew Ng. Deep Learning Specialization. deepleanirng.ai](https://www.coursera.org/specializations/deep-learning?)

## Papers

### Survey

- LeCun Y, Bengio Y, Hinton G. Deep learning[J]. (**Nature 2015**) [[paper]](https://creativecoding.soe.ucsc.edu/courses/cs523/slides/week3/DeepLearning_LeCun.pdf)

- Ruder S. An overview of multi-task learning in deep neural networks[J]. (**arXiv 2017**) [[paper]](https://arxiv.org/pdf/1706.05098.pdf)

- Garcia-Garcia A, Orts-Escolano S, Oprea S, et al. A review on deep learning techniques applied to semantic segmentation[J]. (**arXiv 2017**) [[Segmentation]](https://arxiv.org/pdf/1704.06857.pdf)

- Liu L, Ouyang W, Wang X, et al. Deep learning for generic object detection: A survey[J]. (**arXiv 2018**) [[Detection]](https://arxiv.org/pdf/1809.02165.pdf)

- Hong Y, Hwang U, Yoo J, et al. How Generative Adversarial Networks and Their Variants Work: An Overview[J]. (**CSUR 2019**) [[paper]](https://arxiv.org/pdf/1711.05914.pdf)

### Image Classification

#### Backbone

- Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[C]. (**ICLR 2015**) [[VGGNet]](https://arxiv.org/pdf/1409.1556.pdf)

- Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C]. (**CVPR 2015**) [[Inception]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)

- He K, Zhang X, Ren S, et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification[C]. (**ICCV 2015**) [[PReLU]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)

- Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[C]. (**ICML 2015**) [[BatchNorm]](https://arxiv.org/pdf/1502.03167.pdf)

- Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]. (**CVPR 2016**) [[InceptionV3]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)

- He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]. (**CVPR 2016**) [[ResNet]](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

- He K, Zhang X, Ren S, et al. Identity mappings in deep residual networks[C]. (**ECCV 2016**) [[ResNetV2]](https://arxiv.org/pdf/1603.05027.pdf)

- Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks[C]. (**CVPR 2017**) [[ResNeXt]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf)

- Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional networks[C]. (**CVPR 2017**) [[DenseNet]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)

- Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]. (**CVPR 2018**) [[SE-Net]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)

- Sun S, Pang J, Shi J, et al. FishNet: A versatile backbone for image, region, and pixel level prediction[C]. (**NIPS 2018**) [[FishNet]](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf)

#### Visualization

- Zeiler M D, Fergus R. Visualizing and understanding convolutional networks[C]. (**ECCV 2014**) [[paper]](https://arxiv.org/pdf/1311.2901.pdf)

- Xu K, Ba J, Kiros R, et al. Show, attend and tell: Neural image caption generation with visual attention[C]. (**ICML 2015**) [[paper]](http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf)

- Zhou B, Khosla A, Lapedriza A, et al. Learning deep features for discriminative localization[C]. (**CVPR 2016**) [[CAM]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)

- Geirhos R, Rubisch P, Michaelis C, et al. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness[C]. (**ICLR 2019**) [[shapeVStexture]](https://arxiv.org/pdf/1811.12231.pdf)

#### Cls Others

- Lin M, Chen Q, Yan S. Network in network[C]. (**2013.12**) [[Conv1x1]](https://arxiv.org/pdf/1312.4400.pdf)

- Springenberg J T, Dosovitskiy A, Brox T, et al. Striving for simplicity: The all convolutional net[C] (**ICLR workshop 2015**) [[paper]](https://arxiv.org/pdf/1412.6806.pdf%20(http://arxiv.org/pdf/1412.6806.pdf))

- Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks[C]. (**NIPS 2015**) [[STN]](http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf)

- Yu Q, Wang J, Zhang S, et al. Combining local and global hypotheses in deep neural network for multi-label image classification[J].  (**Neurocomputing 2017**) [[2015.08]](https://pdf.xuebalib.com/xuebalib.com.17720.pdf)

- Chollet F. Xception: Deep learning with depthwise separable convolutions[C]. (**CVPR 2017**) [[Xception]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)

- Howard A G, Zhu M, Chen B, et al. MobileNets: Efficient convolutional neural networks for mobile vision applications[C]. (**2017.04**) [[MobileNet]](https://arxiv.org/pdf/1704.04861.pdf)

- Wu Y, He K. Group normalization[C]. (**ECCV 2018**) [[GroupNorm]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf)

- Xie J, He T, Zhang Z, et al. Bag of tricks for image classification with convolutional neural networks[C]. (**CVPR 2019**) [[tricks]](https://arxiv.org/pdf/1812.01187.pdf)

### Video

- Karpathy A, Toderici G, Shetty S, et al. Large-scale video classification with convolutional neural networks[C]. (**CVPR 2014**) [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Karpathy_Large-scale_Video_Classification_2014_CVPR_paper.pdf)

- Yue-Hei Ng J, Hausknecht M, Vijayanarasimhan S, et al. Beyond short snippets: Deep networks for video classification[C]. (**CVPR 2015**) [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ng_Beyond_Short_Snippets_2015_CVPR_paper.pdf)

- Donahue J, Anne Hendricks L, Guadarrama S, et al. Long-term recurrent convolutional networks for visual recognition and description[C]. (**CVPR 2015**) [[paper]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf)

- Wang X, Girshick R, Gupta A, et al. Non-local neural networks[C]. (**CVPR 2018**) [[Non-local]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)

### Object Detection

#### One-Stage

- Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]. (**CVPR 2016**) [[YOLO]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)

- Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector[C]. (**ECCV 2016**) [[SSD]](https://arxiv.org/pdf/1512.02325.pdf)

- Redmon J, Farhadi A. YOLO9000: better, faster, stronger[C]. (**CVPR 2017**) [[YOLOv2]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)

- Redmon J, Farhadi A. YOLOv3: An incremental improvement[J].  (**arXiv 2018**) [[YOLOv3]](https://arxiv.org/pdf/1804.02767.pdf)

- Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection[C]. (**ICCV 2017**) [[Focal Loss]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

- Zhang S, Wen L, Bian X, et al. Single-shot refinement neural network for object detection[C]. (**CVPR 2018**) [[RefineDet]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Refinement_Neural_CVPR_2018_paper.pdf)

#### Two-Stage

- Girshick R, Donahue J, Darrell T, et al. Rich feature hierarchies for accurate object detection and semantic segmentation[C]. (**CVPR 2014**) [[R-CNN]](http://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)

- Girshick R. Fast R-CNN[C]. (**ICCV 2015**) [[Fast R-CNN]](http://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

- Ren S, He K, Girshick R, et al. Faster r-cnn: Towards real-time object detection with region proposal networks[C]. (**NIPS 2015**) [[Faster R-CNN]](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)

- Shrivastava A, Gupta A, Girshick R. Training region-based object detectors with online hard example mining[C]. (**CVPR 2016**) [[OHEM]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)

- Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection[C]. (**CVPR 2017**) [[FPN]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)

- He K, Gkioxari G, Dollár P, et al. Mask R-CNN[C]. (**ICCV 2017**) [[Mask R-CNN]](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)

### Segmentation

#### Semantic Segmentation

- Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]. (**CVPR 2015**) [[FCN]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

- Chen L C, Papandreou G, Kokkinos I, et al. Semantic image segmentation with deep convolutional nets and fully connected CRFs[J]. (**arXiv 2014**) [[DeepLabV1]](https://arxiv.org/pdf/1412.7062.pdf)

- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation[C]. (**MICCAI 2015**) [[U-Net]](https://arxiv.org/pdf/1505.04597.pdf)

- Badrinarayanan V, Kendall A, Cipolla R. SegNet: A deep convolutional encoder-decoder architecture for image segmentation[J]. (**TPAMI 2017**) [[SegNet]](https://ieeexplore.ieee.org/iel7/34/4359286/07803544.pdf)

- Yu F, Koltun V. Multi-scale context aggregation by dilated convolutions[J]. (**arXiv 2015**) [[paper]](https://arxiv.org/pdf/1511.07122.pdf])

- Wu Z, Shen C, Hengel A. High-performance semantic segmentation using very deep fully convolutional networks[J]. (**arXiv 2016**) [[OHEM]](https://arxiv.org/pdf/1604.04339.pdf)

- Chen L C, Papandreou G, Kokkinos I, et al. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs[J]. (**TPAMI 2018**) [[DeepLabV2]](https://arxiv.org/pdf/1606.00915.pdf)

- Lin G, Milan A, Shen C, et al. RefineNet: Multi-path refinement networks for high-resolution semantic segmentation[C]. (**CVPR 2017**) [[RefineNet]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf)

- Zhao H, Shi J, Qi X, et al. Pyramid scene parsing network[C]. (**CVPR 2017**) [[PSPNet]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)

- Wang P, Chen P, Yuan Y, et al. Understanding convolution for semantic segmentation[C]. (**WACV 2018**) [[DUC]](https://arxiv.org/pdf/1702.08502.pdf)

- Yu F, Koltun V, Funkhouser T. Dilated residual networks[C]. (**CVPR 2017**) [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Dilated_Residual_Networks_CVPR_2017_paper.pdf)

- Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. (**arXiv 2017**) [[DeepLabV3]](https://arxiv.org/pdf/1706.05587.pdf)

- Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]. (**ECCV 2018**) [[DeepLabV3+]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf)

- Ke T W, Hwang J J, Liu Z, et al. Adaptive Affinity Fields for Semantic Segmentation[C]. (**ECCV 2018**) [[AAF]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Jyh-Jing_Hwang_Adaptive_Affinity_Field_ECCV_2018_paper.pdf)

- Zhang H, Dana K, Shi J, et al. Context encoding for semantic segmentation[C]. (**CVPR 2018**) [[EncNet]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Context_Encoding_for_CVPR_2018_paper.pdf)

- Zhang Z, Zhang X, Peng C, et al. ExFuse: Enhancing feature fusion for semantic segmentation[C]. (**ECCV 2018**) [[ExFuse]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenli_Zhang_ExFuse_Enhancing_Feature_ECCV_2018_paper.pdf)

- Yu C, Wang J, Peng C, et al. Learning a discriminative feature network for semantic segmentation[C]. (**CVPR 2018**) [[DFN]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Learning_a_Discriminative_CVPR_2018_paper.pdf)

- Chen L C, Collins M, Zhu Y, et al. Searching for efficient multi-scale architectures for dense image prediction[C]. (**NIPS 2018**) [[NAS Seg]](http://papers.nips.cc/paper/8087-searching-for-efficient-multi-scale-architectures-for-dense-image-prediction.pdf)

- Zhao H, Zhang Y, Liu S, et al. PSANet: Point-wise spatial attention network for scene parsing[C] (**ECCV 2018**) [[PSANet]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.pdf)

- Fu J, Liu J, Tian H, et al. Dual attention network for scene segmentation[J]. (**arXiv 2018**) [[DANet]](https://arxiv.org/pdf/1809.02983.pdf)

- Huang Z, Wang X, Huang L, et al. CCNet: Criss-Cross Attention for Semantic Segmentation[J]. (**arXiv 2018**) [[CCNet]](https://arxiv.org/pdf/1811.11721.pdf)

#### Real time Semantic Segmentation

- Paszke A, Chaurasia A, Kim S, et al. ENet: A deep neural network architecture for real-time semantic segmentation[J]. (**arXiv 2016**) [[ENet]](https://arxiv.org/pdf/1606.02147.pdf)

- Zhao H, Qi X, Shen X, et al. ICNet for real-time semantic segmentation on high-resolution images[C]. (**ECCV 2018**) [[ICNet]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_ICNet_for_Real-Time_ECCV_2018_paper.pdf)

- Mehta S, Rastegari M, Caspi A, et al. ESPnet: Efficient spatial pyramid of dilated convolutions for semantic segmentation[C]. (**ECCV 2018**) [[ESPNet]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sachin_Mehta_ESPNet_Efficient_Spatial_ECCV_2018_paper.pdf)

- Yu C, Wang J, Peng C, et al. BiSeNet: Bilateral segmentation network for real-time semantic segmentation[C] (**ECCV 2018**) [[BiSeNet]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.pdf)

#### Point Cloud

- Qi C R, Su H, Mo K, et al. PointNet: Deep learning on point sets for 3d classification and segmentation[C]. (**CVPR 2017**) [[PointNet]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)

- Qi C R, Yi L, Su H, et al. Pointnet++: Deep hierarchical feature learning on point sets in a metric space[C] (**NIPS 2017**) [[PointNet++]](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf)

#### Instance segmentation

- He K, Gkioxari G, Dollár P, et al. Mask R-CNN[C]. (**ICCV 2017**) [[Mask R-CNN]](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)

#### Panoptic segmentation

- Kirillov A, He K, Girshick R, et al. Panoptic segmentation[J]. (**arXiv 2018**) [[metric]](https://arxiv.org/pdf/1801.00868)

- Yang T J, Collins M D, Zhu Y, et al. DeeperLab: Single-Shot Image Parser[J]. (**arXiv 2019**) [[DeeperLab]](https://arxiv.org/pdf/1902.05093.pdf)

- Kirillov A, Girshick R, He K, et al. Panoptic Feature Pyramid Networks[J]. (**arXiv 2019**) [[Panoptic FPN]](https://arxiv.org/pdf/1901.02446.pdf)

### Domain Adaptation

- Tzeng E, Hoffman J, Zhang N, et al. Deep domain confusion: Maximizing for domain invariance[J]. (**arXiv 2014**) [[paper]](https://arxiv.org/pdf/1412.3474.pdf)

- Ghifary M, Kleijn W B, Zhang M. Domain adaptive neural networks for object recognition[C]. (**PRICAI 2014**) [[paper]](https://arxiv.org/pdf/1409.6041.pdf)

- Long M, Cao Y, Wang J, et al. Learning transferable features with deep adaptation networks[C]. (**ICML 2015**) [[DAN]](http://proceedings.mlr.press/v37/long15.pdf)

- Ganin Y, Lempitsky V. Unsupervised domain adaptation by backpropagation[C]. (**ICML 2015**) [[DaNN]](https://arxiv.org/pdf/1409.7495.pdf)

- Li Y, Wang N, Shi J, et al. Adaptive Batch Normalization for practical domain adaptation[J]. (**PR 2018**) [[AdapaBN]](http://winsty.net/papers/adabn.pdf)

### Nautre Language Process

- Kim Y. Convolutional neural networks for sentence classification[C]. (**EMNLP 2014**) [[TextCNN]](https://arxiv.org/pdf/1408.5882.pdf)[[code]](https://github.com/dennybritz/cnn-text-classification-tf)

- Christopher Olah. Understanding LSTMs. (**2015.08**) [[url]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[C]. (**ICLR 2018**) [[Attention]](https://arxiv.org/pdf/1409.0473.pdf)

- Gehring J, Auli M, Grangier D, et al. Convolutional sequence to sequence learning[C]. (**ICML 2017**) [[ConvS2S]](https://arxiv.org/pdf/1705.03122.pdf)

- Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]. (**NIPS 2017**) [[Transformer]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

### EEG

#### Generic EEG

#### Sleep Stage Classification

## Software and Skills

### Framework

- TensorFlow [[install&docs]](https://www.tensorflow.org/)
- Keras [[docs]](https://keras.io/)
- PyTorch [[install]](http://pytorch.org/) [[docs]](http://pytorch.org/docs/0.3.0/)

### Skills

- [python tutorial](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000)
- [latex](https://tug.org/texlive/)
- [git](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)
