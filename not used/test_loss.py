import tensorflow as tf
import numpy as np
import pdb

batch_size = 20
n_parts = 4

bs = 3 * batch_size
count = 0
# loop over each pair of positive images in the batch to calculate the Net global contrastive loss over the whole batch.
for pos_index in range(0, batch_size, 1):
    # indexes of positive pair of samples (x_1,x_2,x_3) - we can make 3 pairs: (x_1,x_2), (x_1,x_3), (x_2,x_3)
    num_i1 = np.arange(pos_index, pos_index + 1, dtype=np.int32)
    j = batch_size + pos_index
    num_i2 = np.arange(j, j + 1, dtype=np.int32)
    j = 2 * batch_size + pos_index
    num_i3 = np.arange(j, j + 1, dtype=np.int32)
    # print('n1,n2,n3',num_i1,num_i2,num_i3)

    # indexes of corresponding negative samples as per positive pair of samples: (x_1,x_2), (x_1,x_3), (x_2,x_3)
    den_index_net = np.arange(0, bs, dtype=np.int32)

    # Pruning the negative samples
    # Deleting the indexes of the samples in the batch used as negative samples for a given positive image. These indexes belong to identical partitions in other volumes in the batch.
    # Example: if positive image is from partition 1 of volume 1, then NO negative sample are taken from partition 1 of any other volume (including volume 1) in the batch
    ind_l = []
    rem = int(num_i1) % n_parts
    for not_neg_index in range(rem, bs, 4):
        ind_l.append(not_neg_index)

    # print('ind_l',ind_l)
    den_indexes = np.delete(den_index_net, ind_l)
    print("Den indices are:", den_indexes)
    # print('d1',den_i1,len(den_i1))

    # gather required positive samples x_1,x_2,x_3 for the numerator term
    #x_num_i1 = tf.gather(reg_pred, num_i1)
    #x_num_i2 = tf.gather(reg_pred, num_i2)
    #x_num_i3 = tf.gather(reg_pred, num_i3)

    # gather required negative samples x_1,x_2,x_3 for the denominator term
    #x_den = tf.gather(reg_pred, den_indexes)

    # calculate cosine similarity score + global contrastive loss for each pair of positive images

    # for positive pair (x_1,x_2);
    # numerator of loss term (num_i1_i2_ss) & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
    print("num_ss = Cos sim between %d and %d" % (num_i1, num_i2))
    print("den_1_2_ss = Cos sim between %d and %s" % (num_i1, den_indexes))
    print("log(exp(num_ss))/exp(reduce_sum(den_ss))")
    print("Add to loss")
    # for positive pair (x_2,x_1);
    # numerator same & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
    print("din_2_1_ss = Cos sim between %d and %s" % (num_i2, den_indexes))
    print("log(exp(num_ss))/exp(reduce_sum(den_ss_2_1))")
    print("Add to loss")

    # for positive pair (x_1,x_3);
    # numerator of loss term (num_i1_i3_ss) & denominator of loss term (den_i1_i3_ss) & loss (num_i1_i3_loss)
    print("num_ss = Cos sim between %d and %d" % (num_i1, num_i3))
    print("den_1_3_ss = Cos sim between %d and %s" % (num_i1, den_indexes))
    print("log(exp(num_ss))/exp(reduce_sum(den_ss))")
    print("Add to loss")
    # for positive pair (x_3,x_1);
    # numerator same & denominator of loss term (den_i3_i1_ss) & loss (num_i3_i1_loss)
    print("din_3_1_ss = Cos sim between %d and %s" % (num_i3, den_indexes))
    print("log(exp(num_ss))/exp(reduce_sum(den_ss_3_1))")
    print("Add to loss")

    # for positive pair (x_2,x_3);
    # numerator of loss term (num_i2_i3_ss) & denominator of loss term (den_i2_i3_ss) & loss (num_i2_i3_loss)
    print("num_ss = Cos sim between %d and %d" % (num_i2, num_i3))
    print("den_2_3_ss = Cos sim between %d and %s" % (num_i2, den_indexes))
    print("log(exp(num_ss))/exp(reduce_sum(den_ss))")
    print("Add to loss")
    # for positive pair (x_3,x_2):
    # numerator same & denominator of loss term (den_i3_i2_ss) & loss (num_i3_i2_loss)
    print("din_3_2_ss = Cos sim between %d and %s" % (num_i3, den_indexes))
    print("log(exp(num_ss))/exp(reduce_sum(den_ss_3_2))")
    print("Add to loss")
    count = count + 1
    #pdb.set_trace()
print("Done %d" % count)