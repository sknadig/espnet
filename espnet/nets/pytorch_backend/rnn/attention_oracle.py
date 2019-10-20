import math
import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device

import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class OracleAtt(torch.nn.Module):
    def __init__(self, alignment_type="square"):
        super(OracleAtt, self).__init__()
        self.frame_dict = pickle.load(open("/home/shree/espnet/egs/librispeech/asr1/frame_level_fa2.pkl", "rb"))
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
