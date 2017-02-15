#!/bin/bash

echo building vectors
th build_vects.lua

echo randomizing vectors
python randomize_data_file.py

echo training network and predicting new values
th lstm.lua
