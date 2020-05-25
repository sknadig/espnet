#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev
test_set=test

# Set this to one of ["phn", "char"] depending on your requirement
trans_type=phn

asr_config=conf/train_asr.yaml
decode_config=conf/decode_asr.yaml

./asr.sh \
    --asr_tag "triplet_loss_no_sos_after_10e" \
    --token_type ${trans_type} \
    --train_set train \
    --dev_set dev \
    --eval_sets "test " \
    --use_lm false \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --srctexts "data/${train_set}/text" "$@"
