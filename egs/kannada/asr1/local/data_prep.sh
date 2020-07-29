#!/bin/bash

# Copyright 2020   (Authors: Shreekantha Nadig)
# Apache 2.0.

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 downloads/ data/"
  exit 1
fi

src=$1
dst=$2
num_data_reps=3
fs=16000

# all utterances are sox compressed
if ! which sox >&/dev/null; then
   echo "Please install 'sox' on ALL worker nodes!"
   exit 1
fi


for i in male female; do
    mkdir -p ${dst}/${i}
    find ${src}/$i/*.wav > ${dst}/${i}/wav_files
    cp ${src}/${i}/line_index.tsv ${dst}/${i}/text_orig
    cat ${dst}/${i}/text_orig | awk '{print $1}' > ${dst}/${i}/uttids
    paste -d ' ' ${dst}/${i}/uttids <(cat ${dst}/${i}/uttids | cut -d '_' -f1,2) > ${dst}/${i}/utt2spk
    utils/utt2spk_to_spk2utt.pl ${dst}/${i}/utt2spk > ${dst}/${i}/spk2utt
    for wav_file in `cat ${dst}/${i}/wav_files`; do
        echo "sox ${wav_file} -t wav -r ${fs} -b 16 - |" >> ${dst}/${i}/wav_cmd
    done
    paste <(cat ${dst}/${i}/wav_files | cut -d ' ' -f1 | rev | cut -d '/' -f1 | rev | sed -s 's/\.wav//g') ${dst}/${i}/wav_cmd > ${dst}/${i}/wav.scp
    utils/data/get_utt2dur.sh ${dst}/${i}
    dur=$(awk '{ sum += $1 } END { print sum/60 }' <(cat ${dst}/${i}/utt2dur | cut -d ' ' -f2))
    echo "Duration of data in ${dst}/${i} = ${dur} minutes"
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 20 ${dst}/${i} ${dst}/${i}_train ${dst}/${i}_dev
    for s in train dev; do
        dur=$(awk '{ sum += $1 } END { print sum/60 }' <(cat ${dst}/${i}_${s}/utt2dur | cut -d ' ' -f2))
        echo "Duration of data in ${dst}/${i}_${s} = ${dur} minutes"
    done
    python local/split.py ${dst}/${i}/text_orig ${dst}/${i}/text_split
done

utils/combine_data.sh ${dst}/train ${dst}/male_train ${dst}/female_train
utils/data/get_utt2dur.sh ${dst}/train
utils/combine_data.sh ${dst}/dev ${dst}/male_dev ${dst}/female_dev
utils/data/get_utt2dur.sh ${dst}/dev

# Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
if [ ! -f rirs_noises.zip ]; then
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
fi

rvb_opts=()
# This is the config for the system using simulated RIRs and point-source noises
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)

foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"


for s in train dev; do
    utils/copy_data_dir.sh --utt-suffix "-volp" ${dst}/${s} ${dst}/${s}_volp
    utils/data/perturb_data_dir_volume.sh ${dst}/${s}_volp
    utils/data/get_utt2dur.sh ${dst}/${s}_volp
    utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true ${dst}/${s} ${dst}/${s}_sp
    utils/data/get_utt2dur.sh ${dst}/${s}_sp

    # corrupt the data to generate multi-condition data
    rvb_targets_dir=${s}_rvb
    python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix "rev" \
    --foreground-snrs $foreground_snrs \
    --background-snrs $background_snrs \
    --speech-rvb-probability 0.9 \
    --pointsource-noise-addition-probability 0.9 \
    --isotropic-noise-addition-probability 0.9 \
    --num-replications $num_data_reps \
    --max-noises-per-minute 4 \
    --source-sampling-rate ${fs} \
    ${dst}/${s} ${dst}/${s}_rvb

    utils/data/get_utt2dur.sh ${dst}/${s}_rvb

    utils/combine_data.sh ${dst}/${s}_all ${dst}/${s} ${dst}/${s}_volp ${dst}/${s}_sp ${dst}/${s}_rvb
    dur=$(awk '{ sum += $1 } END { print sum/3600 }' <(cat ${dst}/${s}_all/utt2dur | cut -d ' ' -f2))
    echo "Duration of data in ${dst}/${s}_all = ${dur} hours"
done