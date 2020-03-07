#!/bin/bash
#Copyright 2020 IIIT-Bangalore (Shreekantha Nadig)
#Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

#general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
cv_datadir=downloads/cv/
libri_datadir=downloads/libri/
cv_lang=fr # en de fr cy tt kab ca zh-TW it fa eu es ru

# base url for downloads.
cv_data_url=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/${cv_lang}.tar.gz
libri_data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
trans_type=char

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

libri_train_set=libri_train
libri_train_dev=libri_dev
libri_recog_set="libri_test_clean"

cv_train_set=cv_valid_train_${cv_lang}
cv_train_dev=cv_valid_dev_${cv_lang}
cv_test_set=cv_valid_test_${cv_lang}
cv_recog_set="valid_dev_${cv_lang} valid_test_${cv_lang}"

train_set=train_nodev
train_dev=train_dev
recog_set="${cv_test_set} ${libri_recog_set}"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    mkdir -p ${cv_datadir}
    local/commonvoice_download_and_untar.sh ${cv_datadir} ${cv_data_url} ${cv_lang}.tar.gz
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    for part in "validated"; do
        # use underscore-separated names in data directories.
        local/data_prep.pl ${cv_datadir} ${part} data/"$(echo "${part}_${cv_lang}" | tr - _)"
    done

    # Kaldi Version Split
    # ./utils/subset_data_dir_tr_cv.sh data/validated data/valid_train data/valid_test_dev
    # ./utils/subset_data_dir_tr_cv.sh --cv-spk-percent 50 data/valid_test_dev data/valid_test data/valid_dev

    # ESPNet Version (same as voxforge)
    # consider duplicated sentences (does not consider speaker split)
    # filter out the same sentences (also same text) of test&dev set from validated set
    echo data/cv_validated_${cv_lang} data/${cv_train_set} data/${cv_train_dev} data/${cv_test_set}
    local/split_tr_dt_et.sh data/validated_${cv_lang} data/${cv_train_set} data/${cv_train_dev} data/${cv_test_set}

    for part in ${cv_train_set} ${cv_train_dev} ${cv_test_set}; do
        python clean_text.py data/${part}/text data/${part}/text.cleaned
        mv data/${part}/text data/${part}/text.orig
        mv data/${part}/text.cleaned data/${part}/text
        utils/fix_data_dir.sh data/${part}/
    done
    utils/subset_data_dir.sh --speakers data/cv_valid_train_fr/ 75000 data/cv_valid_train_fr_subset
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Data Download"
    for part in dev-clean test-clean train-clean-100; do
        local/download_and_untar.sh ${libri_datadir} ${libri_data_url} ${part}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 2: Data preparation"
    for part in dev-clean test-clean train-clean-100; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${libri_datadir}/LibriSpeech/${part} data/libri_${part//-/_}
    done

    for part in dev-clean test-clean train-clean-100; do
        python clean_text.py data/libri_${part//-/_}/text data/libri_${part//-/_}/text.cleaned
        mv data/libri_${part//-/_}/text data/libri_${part//-/_}/text.orig
        mv data/libri_${part//-/_}/text.cleaned data/libri_${part//-/_}/text

        utils/fix_data_dir.sh data/libri_${part//-/_}/
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Combing data"
    utils/data/combine_data.sh data/train_nodev data/libri_train_clean_100/ data/cv_valid_train_fr_subset/
    utils/data/combine_data.sh data/train_dev data/libri_dev_clean/ data/cv_valid_dev_fr/
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 4: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${train_set} ${train_dev}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 8 --write_utt2num_frames true \
        data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
    data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
    data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
        data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
        ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 5: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text --trans_type ${trans_type} | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type ${trans_type} \
    data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type ${trans_type} \
    data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --trans_type ${trans_type} \
        data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    asr_train.py \
    --config ${train_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --outdir ${expdir}/results \
    --tensorboard-dir tensorboard/${expname} \
    --debugmode ${debugmode} \
    --dict ${dict} \
    --debugdir ${expdir} \
    --minibatches ${N} \
    --verbose ${verbose} \
    --resume ${resume} \
    --train-json ${feat_tr_dir}/data.json \
    --valid-json ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding"
    nj=8
    for rtask in ${recog_set}; do
        (
            decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

            # split data
            splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

            #### use CPU for decoding
            ngpu=0

            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

            score_sclite.sh ${expdir}/${decode_dir} ${dict}

        ) &
    done
    wait
    echo "Finished"
fi
