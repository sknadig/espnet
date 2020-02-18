import math
import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy


import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class OracleAtt(torch.nn.Module):
    def __init__(self, gt_pkl=None, alignment_type="square"):
        super(OracleAtt, self).__init__()
        self.frame_dict = pickle.load(open(gt_pkl, "rb"))
        self.alignment_type = alignment_type
        logging.info("Using alignment type: " + str(alignment_type))

    def gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def __call__(self, e, uttids, output_index):
        e_oracle = torch.ones(e.size()) * -99999
        e_oracle.to("cuda")

        # logging.info(str(e.size()))
        # logging.info(str(uttids))
        for i, utt in enumerate(uttids):
            att_frames = self.frame_dict[utt]
            # logging.info("ORACLE for {utt} is {frames}".format(utt=utt, frames=att_frames))
            if(output_index < len(att_frames)):
                curr_att_frames = att_frames[output_index]
                indices = np.arange(curr_att_frames[0], curr_att_frames[1])
                center = (curr_att_frames[0]+curr_att_frames[1])//2
                if(self.alignment_type == "square"):
                    e_oracle[i][curr_att_frames[0]:curr_att_frames[1]] = float(1)
                elif(self.alignment_type == "impulse"):
                    if(center == curr_att_frames[1]):
                        center = center-1
                    e_oracle[i][center] = float(1)
                elif(self.alignment_type == "gaussian"):
                    e_oracle[i][curr_att_frames[0]:curr_att_frames[1]] = torch.tensor(self.gaussian(indices, center, 1))        
                else:
                    logging.info("ERROR! Please define a correct alignment type")
                    exit()
            else:
                e_oracle[i] = e[i]
        return e_oracle

class OracleSenone(torch.nn.Module):
    def __init__(self, eprojs, odim):
        super(OracleSenone, self).__init__()
        self.frame_dict = pickle.load(open("/home/neo/MS/kaldi/egs/timit/s5/exp/tri3_frame_level_senone_alignment.pkl", "rb"))
        self.senone_output0 = torch.nn.Linear(eprojs, eprojs)
        self.senone_output1 = torch.nn.Linear(eprojs, eprojs)
        self.senone_output2 = torch.nn.Linear(eprojs, odim)

    def get_oracle_senones(self, uttids=None, device=torch.device('cuda')):
        senones = [self.frame_dict[utt] for utt in uttids]
        senones = [np.array(ele) for ele in senones]
        senones = pad_list([torch.from_numpy(np.array(y[0]) if isinstance(y, tuple) else y).long()
                           for y in senones], -1).to(device)
        return senones

    def __call__(self, y, uttids):
        batch = len(uttids)
        olength = y.size(1)
        
        senone_output = self.senone_output0(y)
        senone_output = self.senone_output1(senone_output)
        senone_output = self.senone_output2(senone_output)
        senone_outputs = F.softmax(senone_output, dim=2)
        logging.info("Senone output shape: {0}".format(str(senone_outputs.size())))
        senone_outputs = torch.stack([row for row in senone_outputs], dim=1).view(batch * olength, -1)
        logging.info("Senone output shape: {0}".format(str(senone_outputs.size())))

        oracle_senones = self.get_oracle_senones(uttids=uttids)
        logging.info("ORACLE Senone output shape: {0}".format(str(oracle_senones.size())))
        loss = F.cross_entropy(senone_outputs, oracle_senones.view(-1),
                                    ignore_index=-1,
                                    reduction='mean')
        loss *= (np.mean([len(y) for x in y]) - 1)
        acc = th_accuracy(senone_outputs, oracle_senones, ignore_label=-1)
        return loss, acc