'''
	set all params here
'''

## Neural Network Params
REG_CONST = 0.0001
LEARNING_RATE = .1
MOMENTUM = 0.9

BASE_LAYERS = [ {'filters': 75, 'kernel_size': 4},
				{'filters': 75, 'kernel_size': 4},
				{'filters': 75, 'kernel_size': 4}]


EPSILON = 0.2 # For randomness
ALPHA = 0.8 # For randomness

CPUCT = 1
EPOCHS = 50
BATCH_SIZE = 64
TRAINING_SIZE = 15000

INPUT_SHAPE = (6, 7, 2)
ITERATIONS = 10
GAMES_PER_ITERATION = 1000
H2H_GAMES = 100
NET_THRESHOLD = 0.55

MAX_SIZE_OF_DATASET = 500000

## Condor Params
GAMES_PER_TASK = 5
WORKERS = 0
PORT = 9146
USERNAME = 'jmazane1'
