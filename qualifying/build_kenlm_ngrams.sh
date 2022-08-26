#!/bin/bash

NGRAM=$1
TEXTFILE=$2
OUTPUT=$3

wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
mkdir kenlm/build && cd kenlm/build && cmake .. && make -j8
cd ../../
kenlm/build/bin/lmplz -o "${NGRAM}" <"${TEXTFILE}" > "${OUTPUT}"
