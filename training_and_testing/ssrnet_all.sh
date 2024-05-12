#!/bin/bash

#######################################
# Runs training and plotting for SSRNet
#
# Arguments:
#   database: name of the database to use
#######################################
function main() {
  local db="wiki"
  local current_dir

  # Create the database for training
  echo "Creating database for $db"
  current_dir=$(pwd)
  cd ../data && python3 ../data/TYY_IMDBWIKI_create_db.py --db "$db" --output "$db"_db.npz
  cd "$current_dir" || exit

  # Copy the pre-trained models
  cp -r ../pre-trained/imdb ./imdb_models

  # Train the model
  echo "Training the model for $db"
  KERAS_BACKEND=tensorflow python SSRNET_train.py \
    --input ../data/"$db"_db.npz \
    --db "$db" \
    --netType1 4 \
    --netType2 4 \
    --batch_size 50

  # Plot the training history
  echo "Plotting the training history for $db"
  python plot_reg.py --input "$db"_models/ssrnet_3_3_3_64_1.0_1.0/history_ssrnet_3_3_3_64_1.0_1.0.h5
}

main "$@"
