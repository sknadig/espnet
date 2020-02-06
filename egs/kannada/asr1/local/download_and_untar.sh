#!/bin/bash

# Copyright 2016  Allen Guo
# Copyright 2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base>"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  exit 0;
fi

data=$1
url=$2

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1;
fi

if [ -d $data/an4 ]; then
  echo "$0: kannada directory already exists in $data"
  exit 0;
fi

if [ -f $data/kn_in_female.zip ]; then
  echo "$data/kn_in_female.zip exists"
fi

if [ -f $data/kn_in_male.zip ]; then
  echo "$data/kn_in_male.zip exists"
fi

if [ ! -f $data/kn_in_male.zip ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  full_url=$url/kn_in_male.zip
  transcripts=$url/line_index_male.tsv
  echo "$0: downloading data (840 MB) from $full_url."

  cd $data
  if ! wget --no-check-certificate $full_url; then
    echo "$0: error executing wget $full_url"
    exit 1;
  fi

  if ! wget --no-check-certificate $transcripts; then
    echo "$0: error executing wget $transcripts"
    exit 1;
  fi

fi

if [ ! -f $data/kn_in_female.zip ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  full_url=$url/kn_in_female.zip
  transcripts=$url/line_index_male.tsv
  echo "$0: downloading data (980 MB) from $full_url."

  cd $data
  if ! wget --no-check-certificate $full_url; then
    echo "$0: error executing wget $full_url"
    exit 1;
  fi

  if ! wget --no-check-certificate $transcripts; then
    echo "$0: error executing wget $transcripts"
    exit 1;
  fi

fi

cd $data
mkdir -p male female
unzip kn_in_male.zip -d male
unzip kn_in_female.zip -d female



if $remove_archive; then
  echo "$0: removing $data/kn_in_male.zip file since --remove-archive option was supplied." 
  rm $data/kn_in_male.zip
  echo "$0: removing $data/kn_in_female.zip file since --remove-archive option was supplied."
  rm $data/kn_in_female.zip
fi
