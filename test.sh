#!/usr/bin/env bash

# put run into file with current date/time in format MM-DD_HH:MM:SS.out
python linear_learner.py > out/$(date +"%m-%d_%H:%M:%S").out
