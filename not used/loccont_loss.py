import numpy as np
import torch
import torch.nn as nn
import pdb

class LocalContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(LocalContrastiveLoss, self).__init__()
        self.batch_size = config['batch_size']
        self.n_parts = config['n_parts']
        self.temp_fac = config['temp_fac']
        self.n_dec_blocks = config['n_dec_blocks']
        self.num_filters = len(config["no_filters"])-1
        self.no_local_regions = config["num_local_regions"]
        no_of_neg_local_regions = 5
        self.img_size_x = config["img_size"][0]
        self.img_size_y = config["img_size"][1]
        self.wgt_en = 0

        self.flatten = nn.Flatten()

        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        size_factor = 1 << (self.num_filters - self.n_dec_blocks)
        im_x, im_y = int(self.img_size_x / size_factor) - 4, int(self.img_size_y / size_factor) - 4

        self.pos_sample_indexes = np.zeros((self.no_local_regions, 2), dtype=np.int32)
        self.pos_sample_indexes[0], self.pos_sample_indexes[1], self.pos_sample_indexes[2] = [0, 0], [0,int(im_y / 2)], [0, im_y]
        self.pos_sample_indexes[3], self.pos_sample_indexes[4], self.pos_sample_indexes[5] = [int(im_x / 2), 0], [int(im_x / 2),int(im_y / 2)], [int(im_x / 2),im_y]
        self.pos_sample_indexes[6], self.pos_sample_indexes[7], self.pos_sample_indexes[8] = [im_x, 0], [im_x, int(im_y / 2)], [im_x,im_y]

        # Indexes for negative samples w,r.t a positive sample.
        self.neg_sample_indexes = np.zeros((self.no_local_regions, no_of_neg_local_regions, 2), dtype=np.int32)
        # Each positive local region will have corresponding regions that act as negative samples to be contrasted.
        # For each positive sample, we pick the nearby no_local_regions (5) local regions as negative samples from both the images (x_a1_i, x_a2_i)
        # for local region at (0,0), define the negative samples co-ordinates accordingly
        self.neg_sample_indexes[0, :, :] = [[0, int(im_y / 2)], [int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)],
                                            [0, im_y], [im_x, 0]]
        # similarly, define negative samples co-ordinates according to positive sample
        self.neg_sample_indexes[1, :, :] = [[0, 0], [0, im_y], [int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)],
                                            [int(im_x / 2), im_y]]
        self.neg_sample_indexes[2, :, :] = [[0, 0], [0, int(im_y / 2)], [int(im_x / 2), int(im_y / 2)],
                                            [int(im_x / 2), im_y], [im_x, im_y]]
        self.neg_sample_indexes[3, :, :] = [[0, 0], [0, int(im_y / 2)], [int(im_x / 2), int(im_y / 2)], [im_x, 0],
                                            [im_x, int(im_y / 2)]]
        self.neg_sample_indexes[4, :, :] = [[0, 0], [0, int(im_y / 2)], [int(im_x / 2), 0], [int(im_x / 2), im_y],
                                            [im_x, int(im_y / 2)]]
        self.neg_sample_indexes[5, :, :] = [[0, int(im_y / 2)], [0, im_y], [int(im_x / 2), int(im_y / 2)],
                                            [im_x, int(im_y / 2)], [im_x, im_y]]
        self.neg_sample_indexes[6, :, :] = [[0, 0], [int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)],
                                            [im_x, int(im_y / 2)], [im_x, im_y]]
        self.neg_sample_indexes[7, :, :] = [[int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)], [int(im_x / 2), im_y],
                                            [im_x, 0], [im_x, im_y]]
        self.neg_sample_indexes[8, :, :] = [[0, im_y], [int(im_x / 2), int(im_y / 2)], [int(im_x / 2), im_y], [im_x, 0],
                                            [im_x, int(im_y / 2)]]

    def forward(self, x):
        local_loss = 0
        # loop over each image pair to iterate over all positive local regions within a feature map to calculate the local contrastive loss
        for pos_index in range(0, self.batch_size, 2):

            # indexes of positive pair of samples (f_a1_i,f_a2_i) of input images (x_a1_i,x_a2_i) from the batch of feature maps.
            num_i1 = np.arange(pos_index, pos_index + 1, dtype=np.int32)
            num_i2 = np.arange(pos_index + 1, pos_index + 2, dtype=np.int32)

            # gather required positive samples (f_a1_i,f_a2_i) of (x_a1_i,x_a2_i) for the numerator term
            x_num_i1 = x[num_i1].squeeze()
            x_num_i2 = x[num_i2].squeeze()
            # print('x_num_i1,x_num_i2',x_num_i1,x_num_i2)

            # if local region size is 3x3
            # loop over all defined local regions within a feature map
            for local_pos_index in range(0, self.no_local_regions, 1):
                # 'pos_index_num' is the positive local region index in feature map f_a1_i of image x_a1_i that contributes to the numerator term.
                # fetch x and y coordinates
                pos = self.pos_sample_indexes[local_pos_index]
                x_num_tmp_i1 = x_num_i1[:, pos[0]:pos[0] + 2, pos[1]:pos[1] + 2]
                # x_num_tmp_i1 = torch.gather(x_num_i1, [self.pos_sample_indexes[local_pos_index, 0],
                #                                        self.pos_sample_indexes[local_pos_index, 0] + 1,
                #                                        self.pos_sample_indexes[local_pos_index, 0] + 2], axis=1)
                #
                # x_num_tmp_i1 = torch.gather(x_num_tmp_i1, [self.pos_sample_indexes[local_pos_index, 1],
                #                                            self.pos_sample_indexes[local_pos_index, 1] + 1,
                #                                            self.pos_sample_indexes[local_pos_index, 1] + 2], axis=2)
                x_n_i1_flat = self.flatten(x_num_tmp_i1)
                if (self.wgt_en == 1):
                    x_w3_n_i1 = nn.Linear(x_n_i1_flat.shape[0], 128)(x_n_i1_flat)
                else:
                    x_w3_n_i1 = x_n_i1_flat

                # corresponding positive local region index in feature map f_a2_i of image x_a2_i that contributes to the numerator term.
                # fetch x and y coordinates
                x_num_tmp_i2 = x_num_i2[:, pos[0]:pos[0] + 2, pos[1]:pos[1] + 2]
                # x_num_tmp_i2 = torch.gather(x_num_i2, [self.pos_sample_indexes[local_pos_index, 0],
                #                                        self.pos_sample_indexes[local_pos_index, 0] + 1,
                #                                        self.pos_sample_indexes[local_pos_index, 0] + 2], axis=1)
                # x_num_tmp_i2 = torch.gather(x_num_tmp_i2, [self.pos_sample_indexes[local_pos_index, 1],
                #                                            self.pos_sample_indexes[local_pos_index, 1] + 1,
                #                                            self.pos_sample_indexes[local_pos_index, 1] + 2], axis=2)
                x_n_i2_flat = self.flatten(x_num_tmp_i2)
                if (self.wgt_en == 1):
                    x_w3_n_i2 = nn.Linear(x_n_i2_flat.shape[0], 128)(x_n_i2_flat)
                else:
                    x_w3_n_i2 = x_n_i2_flat

                # calculate cosine similarity score for the pair of positive local regions with index 'pos_index_den' within the feature maps from images (x_a1_i,x_a2_i)
                # loss for positive pairs of local regions in feature maps  (f_a1_i,f_a2_i) & (f_a2_i,f_a1_i) in (num_i1_loss,num_i2_loss)

                # Numerator loss terms of local loss
                num_i1_ss = self.cos_sim(x_w3_n_i1, x_w3_n_i2)/self.temp_fac
                num_i2_ss = self.cos_sim(x_w3_n_i2, x_w3_n_i1)/self.temp_fac

                # Negative local regions as per the chosen positive local region at index 'pos_index_den'
                neg_samples_index_list = np.squeeze(self.neg_sample_indexes[local_pos_index])
                no_of_neg_pts = len(neg_samples_index_list)

                # Denominator loss terms of local loss
                den_i1_ss, den_i2_ss = 0, 0

                for local_neg_index in range(0, no_of_neg_pts, 1):
                    neg_pos = neg_samples_index_list[local_neg_index]
                    # negative local regions in feature map (f_a1_i) from image (x_a1_i)
                    # x_den_tmp_i1 = torch.gather(x_num_i1, [neg_samples_index_list[local_neg_index, 0],
                    #                                        neg_samples_index_list[local_neg_index, 0] + 1,
                    #                                        neg_samples_index_list[local_neg_index, 0] + 2], axis=1)
                    # x_den_tmp_i1 = torch.gather(x_den_tmp_i1, [neg_samples_index_list[local_neg_index, 1],
                    #                                            neg_samples_index_list[local_neg_index, 1] + 1,
                    #                                            neg_samples_index_list[local_neg_index, 1] + 2], axis=2)
                    x_den_tmp_i1 = x_num_i1[:, neg_pos[0]:neg_pos[0] + 2, neg_pos[1]:neg_pos[1] + 2]
                    x_d_i1_flat = self.flatten(x_den_tmp_i1)
                    if (self.wgt_en == 1):
                        x_w3_d_i1 = nn.Linear(x_d_i1_flat.shape[0], 128)(x_d_i1_flat)
                    else:
                        x_w3_d_i1 = x_d_i1_flat

                    # negative local regions in feature map (f_a2_i) from image (x_a2_i)
                    # x_den_tmp_i2 = torch.gather(x_num_i2, [neg_samples_index_list[local_neg_index, 0],
                    #                                        neg_samples_index_list[local_neg_index, 0] + 1,
                    #                                        neg_samples_index_list[local_neg_index, 0] + 2], axis=1)
                    # x_den_tmp_i2 = torch.gather(x_den_tmp_i2, [neg_samples_index_list[local_neg_index, 1],
                    #                                            neg_samples_index_list[local_neg_index, 1] + 1,
                    #                                            neg_samples_index_list[local_neg_index, 1] + 2], axis=2)
                    x_den_tmp_i2 = x_num_i2[:, neg_pos[0]:neg_pos[0] + 2, neg_pos[1]:neg_pos[1] + 2]
                    x_d_i2_flat = self.flatten(x_den_tmp_i2)
                    if (self.wgt_en == 1):
                        x_w3_d_i2 = nn.Linear(x_den_tmp_i2.shape[0], 128)(x_d_i1_flat)
                    else:
                        x_w3_d_i2 = x_d_i2_flat

                    # cosine score b/w local region of feature map (f_a1_i) vs other local regions within the same feature map (f_a1_i)
                    den_i1_ss = den_i1_ss + torch.exp(self.cos_sim(x_w3_n_i1, x_w3_d_i1)/self.temp_fac)
                    # cosine score b/w local region of feature map (f_a1_i) vs other local regions from another feature map (f_a2_i)
                    den_i1_ss = den_i1_ss + torch.exp(self.cos_sim(x_w3_n_i1, x_w3_d_i2)/self.temp_fac)

                    # cosine score b/w local region of feature map (f_a2_i) vs other local regions within the same feature map (f_a2_i)
                    den_i2_ss = den_i2_ss + torch.exp(self.cos_sim(x_w3_n_i2, x_w3_d_i2)/self.temp_fac)
                    # cosine score b/w local region of feature map (f_a2_i) vs other local regions from another feature map (f_a1_i)
                    den_i2_ss = den_i2_ss + torch.exp(self.cos_sim(x_w3_n_i2, x_w3_d_i1)/self.temp_fac)

                # local loss from feature map f_a1_i
                num_i1_loss = -torch.log(torch.sum(torch.exp(num_i1_ss)) / (
                        torch.sum(torch.exp(num_i1_ss)) + torch.sum(den_i1_ss)))
                # num_i1_loss=-torch.log(torch.exp(num_i1_ss)/(torch.exp(num_i1_ss)+torch.sum(den_i1_ss)))
                local_loss = local_loss + num_i1_loss

                # local loss from feature map f_a2_i
                num_i2_loss = -torch.log(torch.sum(torch.exp(num_i2_ss)) / (
                        torch.sum(torch.exp(num_i2_ss)) + torch.sum(den_i2_ss)))
                # num_i2_loss=-torch.log(torch.exp(num_i2_ss)/(torch.exp(num_i2_ss)+torch.sum(den_i2_ss)))
                local_loss = local_loss + num_i2_loss

        return local_loss/self.batch_size/self.no_local_regions
