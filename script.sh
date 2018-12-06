#!/bin/bash

module load tensorflow/1.10

python cloud/playGamesVsSelf.py cloud/models/50gamesPer.h5 "$1" "$2"
