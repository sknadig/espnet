#!/usr/bin/env python

# Copyright 2019 IIIT-Bangalore (Shreekantha Nadig)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import configargparse
import json
import sys

def get_parser():
    parser = configargparse.ArgumentParser(
            description="Merge 2 data.${trans_type}.json ",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
        # general configuration
    parser.add_argument('--phn-json', type=str, default=None,
                        help='Filename of phn data (json)')
    parser.add_argument('--senone-json', type=str, default=None,
                        help='Filename of senone data (json)')
    parser.add_argument('--char-json', type=str, default=None,
                        help='Filename of char data (json)')
    parser.add_argument('--out-json', type=str, default=None,
                        help='Filename of out data (json)')
    return parser

def main(cmd_args):
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    
    senone_json = json.load(open(args.senone_json, "r"))
    phn_json = json.load(open(args.phn_json, "r"))
    char_json = json.load(open(args.char_json, "r"))
    out_json = senone_json
    
    print("senone json is: ", args.senone_json)
    print("phn json is: ", args.phn_json)
    print("char json is: ", args.char_json)
    print("out json is: ", args.out_json)

    for uttid in out_json["utts"]:
        out_json["utts"][uttid]["output"][0]["name"] = "target0"

    for uttid in phn_json["utts"]:
        phn_json["utts"][uttid]["output"][0]["name"] = "target1"

    for uttid in char_json["utts"]:
        char_json["utts"][uttid]["output"][0]["name"] = "target2"
    
    for uttid in senone_json["utts"]:
        out_json["utts"][uttid]["output"].append(phn_json["utts"][uttid]["output"][0])
        out_json["utts"][uttid]["output"].append(char_json["utts"][uttid]["output"][0])
    
    with open(args.out_json, "w") as json_out:
        json_out.write(json.dumps(out_json, sort_keys=True, indent=4))
    # json.dump(out_json, open(args.out_json, "w"))

if __name__ == '__main__':
    main(sys.argv[1:])
