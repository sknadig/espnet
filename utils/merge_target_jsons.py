#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import json
import logging
import os
import sys

from espnet.utils.cli_utils import get_commandline_args

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(
        description='merge json files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-jsons', type=str, nargs='+', action='append',
                        default=[], help='Json files for the inputs')
    parser.add_argument('--output-json', dest='output', type=str, help='Output json file')
    return parser

def change_target_id(output, target_id):
    output[0]['name'] = 'target'+str(target_id)
    return output

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"   
    logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    target_jsons = []
    target_keys = []

    for target_json in args.input_jsons[0]:
        print(target_json)
        json_data = json.load(open(target_json, 'r'))
        target_jsons.append(json_data)
        target_keys.append(json_data['utts'].keys())
    
    out_json = {'utts' : {}}

    for key in list(target_keys[0]):
        for target_id, target_json in enumerate(target_jsons):
            if key not in list(out_json['utts'].keys()):
                out_json['utts'][key] = {}
            out_json['utts'][key]['input'] = target_json['utts'][key]['input']
            if 'output' not in list(out_json['utts'][key].keys()):
                out_json['utts'][key]['output'] = []
            out_json['utts'][key]['output'].append(change_target_id(target_json['utts'][key]['output'], target_id))
    
    with open(args.output, 'w') as f:
        f.write(json.dumps(out_json, indent=4, sort_keys=True))