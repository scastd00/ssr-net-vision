#!/bin/bash

#######################################
# Install all the required dependencies
# Arguments:
#  None
#######################################
function main() {
  pip install --upgrade pip
  pip install virtualenv
  python3 -m venv ./.venv
  source ./.venv/bin/activate
  pip install \
  	tensorflow \
    matplotlib \
    pydot \
    graphviz \
    numpy==1.24.3 \
    pandas \
    dlib \
    pygame \
    requests \
    opencv-python \
    moviepy \
    mtcnn \
    scikit-learn \
    scipy \
    tqdm \
    tables

  echo "All dependencies installed successfully"
}

main "$@"
