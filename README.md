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

The USZ Tumor/Metastases dataset is private.


 
All the images were bias corrected using N4 algorithm with a threshold value of 0.001. For more details, refer to the "N4_bias_correction.py" file in scripts.<br/>
Image and label pairs are re-sampled (to chosen target resolution) and cropped/zero-padded to a fixed size using "create_cropped_imgs.py" file. <br/>

IV) Train the models.<br/>
Below commands are an example for ACDC dataset.<br/> 
The models need to be trained sequentially as follows (check "train_model/pretrain_and_fine_tune_script.sh" script for commands)<br/>
Steps :<br/>
1) Step 1: To pre-train the encoder with global loss by incorporating proposed domain knowledge when defining positive and negative pairs.<br/>
cd train_model/ <br/>
python pretr_encoder_global_contrastive_loss.py --dataset=acdc --no_of_tr_imgs=tr52 --global_loss_exp_no=2 --n_parts=4 --temp_fac=0.1 --bt_size=12

2) Step 2: After step 1, we pre-train the decoder with proposed local loss to aid segmentation task by learning distinctive local-level representations.<br/>
python pretr_decoder_local_contrastive_loss.py --dataset=acdc --no_of_tr_imgs=tr52 --pretr_no_of_tr_imgs=tr52 --local_reg_size=1 --no_of_local_regions=13 --temp_fac=0.1 --global_loss_exp_no=2 --local_loss_exp_no=0 --no_of_decoder_blocks=3 --no_of_neg_local_regions=5 --bt_size=12

3) Step 3: We use the pre-trained encoder and decoder weights as initialization and fine-tune to segmentation task using limited annotations.<br/>
python ft_pretr_encoder_decoder_net_local_loss.py --dataset=acdc --pretr_no_of_tr_imgs=tr52 --local_reg_size=1 --no_of_local_regions=13 --temp_fac=0.1 --global_loss_exp_no=2 --local_loss_exp_no=0 --no_of_decoder_blocks=3 --no_of_neg_local_regions=5 --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --ver=0 

To train the baseline with affine and random deformations & intensity transformations for comparison, use the below code file.<br/>
cd train_model/ <br/>
python tr_baseline.py --dataset=acdc --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --ver=0

V) Config files contents.<br/>
One can modify the contents of the below 2 config files to run the required experiments.<br/>
experiment_init directory contains 2 files.<br/>
Example for ACDC dataset:<br/>
1) init_acdc.py <br/>
--> contains the config details like target resolution, image dimensions, data path where the dataset is stored and path to save the trained models.<br/>
2) data_cfg_acdc.py <br/>
--> contains an example of data config details where one can set the patient ids which they want to use as train, validation and test images.<br/>

