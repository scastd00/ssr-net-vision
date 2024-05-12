#!/bin/bash

db="imdb"

echo "Creating database for $db"
current_dir=$(pwd)
cd ../data && python3 ../data/Evaluation_create_db.py --db "$db" --output "$db"_db.npz
cd "$current_dir" || exit

KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python SSRNET_Evaluation.py
