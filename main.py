#!/usr/bin/env python

import numpy as np

from games.connect4 import Connect4
from nn import NN

def main():
	## Intialize variables
	input_shape = (43,)
	iterations = 5
	training_examples = list()
	games_per_iter = 100

	head_to_head_games = 100
	threshold = 0.55

	## Create basic neural network
	nn = NN(input_shape)

	## Initialize the game
	game = Connect4()

	for it in range(iterations):
		# Here, we need to:
		#	1.) Play a bunch of games
		#	2.) Format each game state in the form of a
		# 		training example
		# 	3.) Add them to training examples
		#	4.) Eventually this should be migrated to use WorkQueue,
		#		but not until we get it to work
		for g in range(games_per_iter):
			training_examples.extend(playGameVsSelf(game, nn))

		# After we collect enough examples, we need to train a new
		# Neural network on those examples
		# Once we get this working, we should look into using AWS to train
		# distributed (assuming that we cant train using CRC)
		new_nn = NN(input_shape)
		#new_nn.train(training_examples)

		# Finally, new_nn plays nn and the better of the two becomes nn
		wins = 0
		for g in range(head_to_head_games):
			wins += playGameVsNN(game, nn, new_nn)
		if wins/float(head_to_head_games) >= threshold:
			nn = new_nn 

def playGameVsSelf(game, nn):
	return []

def playGameVsNN(game, nn1, nn2):
	return 1

def test_game(game):
	s = game.startState()
	while True:
		a = game.getValidActions(s)
		s = game.nextState(s, a[0])
		game.printState(s)
		print ''
		t = game.gameOver(s)
		if t:
			print "{} wins".format(t)
			break

	print(s)

def test_net(nn):
	## Init variables
	batch_size = 128
	epochs = 10

	## Create dumby data
	x_train = np.random.randint(high=1, low=-1, size=(1000,nn.input_shape[0]))
	y_train_p = np.random.randint(10, size=(1000, 42))
	y_train_v = np.random.randint(10, size=(1000, 1))
	
	x_test = np.random.randint(high=1, low=-1, size=(100,nn.input_shape[0]))
	y_test_p = np.random.randint(10, size=(100, 42))
	y_test_v = np.random.randint(10, size=(100, 1))

	## Train model
	nn.fit(x_train, [y_train_p, y_train_v], epochs, batch_size)

	## Test model
	score = nn.evaluate(x_test, [y_test_p, y_test_v], batch_size)
	print("Score: {}".format(score))


if __name__ == "__main__":
	main()
