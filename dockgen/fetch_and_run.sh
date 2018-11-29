#!/bin/bash

## Clone the correct github repo

### COPY FILES FROM S3 THAT WE NEED
# nn.py, mcts.py, agent.py, games/connect4.py, playGamesVsSelf.py, params.py
git clone https://github.com/jmazanec15/CloudComputingProject.git
cd CloudComputingProject

### 


## Run the self play game 
# playGamesVsSelf.py path_to_model num_games worker_num
python playGamesVsSelf.py "$1" "$2" "$3" 
