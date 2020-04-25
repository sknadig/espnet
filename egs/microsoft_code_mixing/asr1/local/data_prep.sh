#!/bin/bash

# Copyright 2020  Shreekantha Nadig
# Apache 2.0

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /home/neo/MS/IS_20/ PartB_Gujarati Train data/ "
  exit 1
fi

src=$1
lang=$2
split=$3
data=$4
dst=$data/$lang/$split

mkdir -p $dst || exit 1

wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk

local/collapse_labels.py ${src}/${lang}/${split}/Transcription_LT_Sequence.tsv ${src}/${lang}/${split}/Transcription_LT_Sequence_new.tsv
cat ${src}/${lang}/${split}/Transcription_LT_Sequence_new.tsv | awk '{print $1}' > ${dst}/uttids

paste ${dst}/uttids ${dst}/uttids > ${dst}/utt2spk

for i in `cat ${src}/${lang}/${split}/Transcription_LT_Sequence_new.tsv | awk '{print $1}'`; do echo "$i sox ${src}/${lang}/${split}/Audio/$i.wav -t wav -r 16000 -b 16 - |" >> ${dst}/wav.scp; done;

for i in `cat ${src}/${lang}/${split}/Transcription_LT_Sequence_new.tsv | awk '{print $2}'`; do echo "$i" >> ${dst}/trans; done;

paste ${dst}/uttids ${dst}/trans > ${dst}/text

utils/utt2spk_to_spk2utt.pl ${dst}/utt2spk > ${dst}/spk2utt