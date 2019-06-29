import yaml                                                                                                                                                   │··············
etypes = ["blstmp", "bgrup", "vggblstmp", "vggbgrup"]                                                                                                         │··············
elayers = [1,3,5]                                                                                                                                             │··············
atypes = ["dot", "add", "location", "coverage", "coverage_location", "location2d", "location_recurrent","multi_head_dot", "multi_head_add", "multi_head_loc","│··············
multi_head_multi_res_loc"]                                                                                                                                    │··············
dtypes = ["lstm", "gru"]                                                                                                                                      │··············
                                                                                                                                                              │··············
experiment_id = 0                                                                                                                                             │··············
for atype in atypes:                                                                                                                                          │··············
    for etype in etypes:                                                                                                                                      │··············
        for elayer in elayers:                                                                                                                                │··············
            for dtype in dtypes:                                                                                                                              │··············
                base_data = yaml.load(open("conf/train.yaml", "r"), Loader=yaml.Loader)                                                                       │··············
                base_data["atype"] = atype                                                                                                                    │··············
                base_data["etype"] = etype                                                                                                                    │··············
                base_data["elayers"] = elayer                                                                                                                 │··············
                base_data["dtype"] = dtype                                                                                                                    │··············
                base_data["tap-enc-phn"] = elayer                                                                                                             │··············
                yaml.dump(base_data, open("conf/{0}.yaml".format(str(experiment_id).zfill(3)), "w"), default_flow_style=False)                                │··············
                experiment_cmd = "CUDA_VISIBLE_DEVICES=0,1 ./run.sh --train_config conf/{0}.yaml".format(str(experiment_id).zfill(3))                         │··············
                if(experiment_id < 66):                                                                                                                       │··············
                    with open("exp1.sh", "a") as f:                                                                                                           │··············
                        f.write(experiment_cmd + "\n")                                                                                                        │··············
                elif(experiment_id < 132):                                                                                                                    │··············
                    with open("exp2.sh", "a") as f:                                                                                                           │··············
                        f.write(experiment_cmd + "\n")                                                                                                        │··············
                elif(experiment_id < 198):                                                                                                                    │··············
                    with open("exp3.sh", "a") as f:                                                                                                           │··············
                        f.write(experiment_cmd + "\n")                                                                                                        │··············
                else:                                                                                                                                         │··············
                    with open("exp4.sh", "a") as f:                                                                                                           │··············
                        f.write(experiment_cmd + "\n")                                                                                                        │··············
                experiment_id += 1 