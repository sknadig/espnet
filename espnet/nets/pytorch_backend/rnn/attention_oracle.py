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
    def __init__(self):
        super(OracleAtt, self).__init__()
        self.frame_dict = pickle.load(open("/home/shree/espnet/egs/timit/asr1/frame_level_dict.pkl", "rb"))
    def __call__(self, e, uttids, output_index):
        # logging.info("Normal e size: " + str(e.size()))
        e_oracle = torch.ones(e.size()) * -99999
        #e_oracle = torch.zeros(e.size()).to("cuda")
        e_oracle.to("cuda")
        # for i in range(len(e_oracle)):
        #     e_oracle[i] = float(min(e[i]))
        for i, utt in enumerate(uttids):
            att_frames = self.frame_dict[utt]
            # logging.info("ORACLE for {utt} is {frames}".format(utt=utt, frames=att_frames))
            if(output_index < len(att_frames)):
                curr_att_frames = att_frames[output_index]
                # e_    oracle[i] = float(min(e[i]))
                e_oracle[i][curr_att_frames[0]:curr_att_frames[1]] = float(1)
            else:
                e_oracle[i] = e[i]
            # logging.info("Min e[{0}] is {1}".format(str(i), str(min(e[i]))))
            # logging.info("Full e[{0}] = {1}".format(str(i), str(e_oracle[i])))
            # logging.info("Current att frames for {utt} at output index {idx} are {frames}".format(utt=utt, idx=output_index, frames=curr_att_frames))
        return e_oracle
