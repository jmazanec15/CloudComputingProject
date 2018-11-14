#!/usr/bin/env python

import numpy as np

from games.connect4 import Connect4
from neuralNet.nn import NN

def main():
	## Intialize variables
	input_shape = (43,)

	## Create neural network
	nn = NN(input_shape)
	test_net(nn)

	## Play connect 4
	game = Connect4()
	test_game(game)


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