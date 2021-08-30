import numpy as np
import torch
import torch.nn as nn
import pdb

class GlobalLossDminus(nn.Module):
    def __init__(self, config_encoder, within_dataset = False):
        super(GlobalLossDminus, self).__init__()
        # if dtype == 'train' :
        #    self.batch_size = config_encoder['batch_size']
        # else :
        #    self.batch_size = config_encoder['validation_batch_size']

        self.n_parts = config_encoder['n_parts']
        self.n_datasets = len(config_encoder['name_dataset'])
        self.n_volumes = config_encoder['n_volumes']
        self.n_transforms = config_encoder['n_transforms']
        self.batch_size = self.n_parts * self.n_datasets * self.n_volumes 
        self.weight_loss = config_encoder['weight_loss']
        self.temp_fac = config_encoder['temp_fac']
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.within_dataset = within_dataset
        self.dataset_similar = config_encoder['dataset_similar']

    def compute_symmetrical_loss(self, xi, xj, x_den):
        loss_tmp = 0
        
        if len(xi.shape) > 2 :
            xi = xi.view(xi.shape[0], -1)
            xj = xj.view(xj.shape[0], -1)
            x_den = x_den.view(x_den.shape[0], -1)

        num_i1_i2_ss = self.cos_sim(xi, xj) / self.temp_fac

        # for positive pair (x_1,x_2);
        den_i1_i2_ss = self.cos_sim(xi, x_den) / self.temp_fac
        if self.weight_loss > 1 :
            num_i1_i2_loss = -torch.log(torch.exp(num_i1_i2_ss) /
                                        (torch.exp(num_i1_i2_ss) +
                                         torch.sum( dtype=torch.float32, device=xi.device), torch.exp(den_i1_i2_ss)))
        else :
            num_i1_i2_loss = -torch.log(torch.exp(num_i1_i2_ss) / 
                                        (torch.exp(num_i1_i2_ss) + 
                                         torch.sum(torch.exp(den_i1_i2_ss))))

        loss_tmp = loss_tmp + num_i1_i2_loss

        # for positive pair (x_2,x_1);
        # numerator same & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
        den_i2_i1_ss = self.cos_sim(xj, x_den) / self.temp_fac
        if self.weight_loss > 1 :
            num_i2_i1_loss = -torch.log(torch.exp(num_i1_i2_ss) /
                                        (torch.exp(num_i1_i2_ss) +
                                         torch.sum(dtype=torch.float32, device=xi.device), torch.exp(den_i2_i1_ss)))
        else :
            num_i2_i1_loss = -torch.log(torch.exp(num_i1_i2_ss) / 
                                        (torch.exp(num_i1_i2_ss) + 
                                         torch.sum(torch.exp(den_i2_i1_ss))))

        loss_tmp = loss_tmp + num_i2_i1_loss

        return loss_tmp

    def forward(self, reg_pred):

        n_parts = self.n_parts
        n_datasets = self.n_datasets
        bs = self.batch_size

        net_global_loss = torch.zeros(1, device=reg_pred.device)

        for pos_index in range(0, bs, 1):
            # indexes of positive pair of samples (x_1,x_2,x_3) - we can make 3 pairs: (x_1,x_2), (x_1,x_3), (x_2,x_3)
            num_i1 = np.arange(pos_index, pos_index + 1, dtype=np.int32)
            
            j = bs + pos_index
            num_i2 = np.arange(j, j + 1, dtype=np.int32)
            
            if self.n_transforms == 2:
                j = self.n_transforms * bs + pos_index
                num_i3 = np.arange(j, j + 1, dtype=np.int32)

            # the current positive pair belongs to the dataset with index (num_i1 // n_parts) % n_datasets
            dataset = (num_i1 // n_parts) % n_datasets

            # indexes of corresponding negative samples as per positive pair of samples.
            den_index_net = np.arange(0, bs*(self.n_transforms + 1), dtype=np.int32)

            ind_l = []
            # similar partition of same dataset starts at index rem
            rem = int(num_i1) % (n_parts * n_datasets)

            # add to the list every similar partition of same dataset
            if self.dataset_similar :
                for not_neg_index in range(rem, bs*(self.n_transforms + 1), n_parts):
                    ind_l.append(not_neg_index)
            else :
                for not_neg_index in range(rem, bs*(self.n_transforms + 1), n_parts * n_datasets):
                    ind_l.append(not_neg_index)
            
            if self.within_dataset :
                for i in range(bs * (self.n_transforms + 1)) :
                    if ((i // n_parts) % n_datasets) != ((num_i1 // n_parts) % n_datasets) :
                        ind_l.append(i)

            # remove those similar images from the negative pairs
            den_indexes = np.delete(den_index_net, ind_l)
            
            # gather required positive samples x_1,x_2,x_3 for the numerator term
            x_num_i1 = reg_pred[num_i1]
            x_num_i2 = reg_pred[num_i2]
            if self.n_transforms == 2:
                x_num_i3 = reg_pred[num_i3]

            # gather required negative samples x_1,x_2,x_3 for the denominator term
            x_den = reg_pred[den_indexes]

            # calculate cosine similarity score + global contrastive loss for each pair of positive images
            # for positive pair (x_1,x_2) and for positive pair (x_2,x_1)
            net_global_loss += self.compute_symmetrical_loss(x_num_i1, x_num_i2, x_den)

            if self.n_transforms == 2:
                # for positive pair (x_1,x_3) and for positive pair (x_1,x_3)
                net_global_loss += self.compute_symmetrical_loss(x_num_i1, x_num_i3, x_den)

                # for positive pair (x_2,x_3) and for positive pair (x_3,x_2)
                net_global_loss += self.compute_symmetrical_loss(x_num_i2, x_num_i3, x_den)

            
        net_global_loss /= ((self.n_transforms) * self.batch_size) 
        return net_global_loss.mean()
    
        print('pos_index : ', pos_index)
        print('j : ', j)
        print('num_i1 : ', num_i1)
        print('num_i2 : ', num_i2)
        if self.n_transforms == 2:
            print('num_i3 : ', num_i3)
        print('rem : ', rem)
        print('ind_l',ind_l)
        print('den_indexes',den_indexes)
        print('x_num_i1 shape : ', x_num_i1.shape )
        print('x_num_i2 shape : ', x_num_i2.shape )
        print('x_den shape : ', x_den.shape )
