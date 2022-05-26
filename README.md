**Learning a Universal Encoder for Medical Dataset using Self-Supervised Contrastive Learning** <br/>

The code is for the Master Thesis "Learning a Universal Encoder for Medical Dataset using Self-Supervised Contrastive Learning". 
This project aims to learn a universal encoder for medical images using Self-Supervised learning that could be used with transfer learning for specific segmentation tasks using limited amount of annotations. To do so, several encoders were pre-trained using Contrastive learning and several datasets.

The assessment of the quality of the transfer learning was carried as follows. 
First, the encoder was initialized with a pre-trained network that has been self-trained with only images of the downstream-task. Each segmentation task was thus fine-tuned with its specific pre-trained encoder.
Then the network was initialized with an anatomy-specific encoder that was pre-trained with multiple datasets of the same type of anatomy, including the target one. In order to study whether a more general encoder can learn global representations and be useful for the segmentation of variable anatomies, the network was then initialized with an encoder that has been pre-trained with 4 different datasets each presenting a different anatomy or modality and including the target dataset. Finally, the fine-tuning was done on an encoder that was trained with 4 different datasets presenting different anatomies and excluding the target dataset to assess transferability of the universal encoder.

**Observations / Conclusions:** <br/>
1) On each dataset, this pre-training method showed a consistent benefit on segmentations tasks. 
2) Enriching the pre-training images by several datasets of the same or different anatomies led to an improvement of the results on the secondary tasks compared to the use of an isolated dataset. 
3) In addition, the transferability of these encoders was tested on datasets different from those seen in pre-training to assess the agility of the encoders to generalize the learned representations. 
4) Universal encoders have shown greater ability to match the performance of encoders trained with target datasets and better generalization capabilities. 
5) As a last step, the method was applied to a more tedious task, the segmentation of brain lesions. The experiments, which are currently incomplete, shown encouraging results for the improvement of the segmentation of brain tumors and metastases.

**Authors:** <br/>
Camille Elleaume ([email](mailto:ca.elleaume@gmail.com)),<br/>

**Supervisors:** <br/>
Krishna Chaitanya ([email](mailto:krishna.chaitanya@vision.ee.ethz.ch)),<br/>
Ertunc Erdil,<br/>
Ender Konukoglu.<br/>

**Requirements:** <br/>
Python 3.6.1,<br/>
Tensorflow 1.12.0,<br/>
rest of the requirements are mentioned in the "requirements.txt" file. <br/>

I)  To clone the git repository.<br/>
git clone https://github.com/Elleaume/PDM.git <br/>

II) Install python, required packages and tensorflow.<br/>
Then, install python packages required using below command or the packages mentioned in the file.<br/>
pip install -r requirements.txt <br/>

III) Dataset download.<br/>

To download the ACDC Cardiac dataset, check the website :<br/>
https://www.creatis.insa-lyon.fr/Challenge/acdc. <br/>

To download the MMWHS Cardiac dataset, check the website :<br/>
http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/

To download the CIMAS Cardiac dataset, check the website :<br/>
http://www.cardiacatlas.org/challenges/lv-segmentation-challenge/

To download the ABIDE Brain dataset, check the website :<br/>
http://fcon_1000.projects.nitrc.org/indi/abide/

To download the HCP Brain dataset, check the website :<br/> 
https://www.humanconnectome.org/study/hcp-young-adult/data-releases

To download the Medical Decathlon Prostate dataset, check the website :<br/>
http://medicaldecathlon.com/

To download the CHAOS Abdomens dataset, check the website :<br/> 
https://chaos.grand-challenge.org/Data/

The USZ Tumor/Metastases dataset is private.<br/> 

IV) Complete config files<br/>
The file preprocessing_datasets.json contains parameter for the preprocessing of each dataset<br/>
The config_encoder.json file contains the information needed for pre-training. It's important to choose here which dataset will be use for pre-training, how many partitions and with which loss. The valid dataset names in this list must be included among the datasets listed in the preprocessing_datasets.json file. The path of the images file must also be indicated in this file.<br/>
The seg_unet_\*.json file contains the parameters for the network architecture depending on the dataset segmented. <br/>
Pay attention to the save directory path in each of those file to keep track of your results. <br/>

V) Preprocessing<br/>
Preprocessing is performed with the Preprocessing_with_mask.ipynb Jupyter notebook.<br/>
All the images were bias corrected using N4 algorithm with a threshold value of 0.001. <br/>
Image and label pairs are re-sampled (to chosen target resolution) and cropped/zero-padded to a fixed size. <br/>
If you want to include new datasets you have to fill in their parameters in the preprocessing_datasets.json file.<br/>

VI) Pre-train the models <br/>
Once the config files have been completed, pre-train the model by running the pretraining_FullTrainSet.py file <br/>
Use the command line : python3 pretraining_FullTrainSet.py --config configs/encoder_pretrain.json

IV) Train the models.<br/>
Train the model by running the training.py file. Chose carefully the config file used line 36. This file will determine which dataste is segmented.
Use the command line : python3 training.py --config configs/encoder_pretrain.json

For more information you can refer to the pdf report: MasterThesis_CamilleElleaume.pdf<br/>
All results presented can be reproduced with results\*.ipynb files.
