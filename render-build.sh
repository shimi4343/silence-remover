#!/usr/bin/env bash
set -o errexit

# Install libsndfile for soundfile package
apt-get update && apt-get install -y libsndfile1

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
