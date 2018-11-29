#!/usr/bin/env python

import numpy as np

from connect4 import Connect4
from nn import NN
from agent import ComputerAgent, HumanAgent

def main():
	'''
		This function should generate a trained model that will allow us
		to play the games
	'''
	### Parameters
	input_shape = (6, 7, 1)
	iterations = 10
	games_per_iter = 100
	head_to_head_games = 50
	threshold = 0.55
	cpuct = 1
	epochs = 10
	batch_size = 64
	training_size = 10000

	### Setup variables
	nn = NN(input_shape)
	nn.save_model('./models/v1.h5')
	game = Connect4()
	training_examples = np.array([])
	policies = list()
	values = list()

	for it in range(iterations):
		print("ITERATION: {}/{}".format(it+1, iterations))
		
		### Self playing to accumulate training examples
		print("Self playing {} games".format(games_per_iter))
		for g in range(games_per_iter):
			new_examples, ps, vs = playGameSelfVsSelf(game, nn, cpuct)
			if len(training_examples) != 0:
				training_examples = np.append(training_examples, new_examples, axis=0)
				policies = np.append(policies, ps, axis=0)
				values = np.append(values, vs, axis=0)
			else:	
				training_examples = new_examples
				policies = ps
				values = vs


		### Train new neural net with random assortment of new examples
		print("Training...")
		new_nn = NN(input_shape)

		### Take random subset of the data for training
		size = min(len(training_examples), training_size)
		indices = np.random.choice(range(len(training_examples)), size=size, replace=False)
		ex_subset = training_examples[indices,:]
		p_subset = policies[indices]
		v_subset = values[indices]

		### Train the new neural network
		new_nn.fit(ex_subset, [p_subset, v_subset], epochs, batch_size)

		### Have the nets play each other and the best one survives
		print("Head to head...")
		wins = 0
		for g in range(head_to_head_games):
			if playGameNN1VsNN2(game, nn, new_nn, cpuct) > 0:
				wins+=1
		
		if wins/float(head_to_head_games) >= threshold:
			nn = new_nn 

	#playGameHumanVsComp(game, nn, cpuct)

	nn.save_model('./models/v1.h5')


def playGameHumanVsComp(game, nn, cpuct, first=True):
	if first:
		a1 = ComputerAgent(game, cpuct, nn)
		a2 = HumanAgent(game)
	else:
		a1 = HumanAgent(game)
		a2 = ComputerAgent(game, cpuct, nn)

	return playGame(game, a1, a2)

def playGameSelfVsSelf(game, nn, cpuct, first=True):
	'''
		Play games against itself, with a degree of randomness and collect the results

		For each state, return the state and return the result
	'''
	a1 = ComputerAgent(game, cpuct, nn)
	a2 = ComputerAgent(game, cpuct, nn)
	return playGameSim(game, a1, a2)

def playGameNN1VsNN2(game, nn1, nn2, cpuct, first=True):
	if first:
		a1 = ComputerAgent(game, cpuct, nn1)
		a2 = ComputerAgent(game, cpuct, nn2)
	else:
		a1 = ComputerAgent(game, cpuct, nn2)
		a2 = ComputerAgent(game, cpuct, nn1)

	return playGame(game, a1, a2, output=False)


def playGameSim(game, a1, a2):
	s = game.startState()

	examples = list()
	ps = list()
	vs = list()

	a1_turn = True
	winner = game.gameOver(s)

	while not winner:
		if a1_turn:
			a, p = a1.getMove(s, get_policy=True)
			a1_turn = False
		else:
			a, p = a2.getMove(s, get_policy=True)
			a1_turn = True

		examples.append(s)
		ps.append(p)

		s = game.nextState(s, a)
		winner = game.gameOver(s)


	for _ in ps:
		vs.append(winner)

	return np.array(examples), np.array(ps), np.array(vs)


def playGame(game, a1, a2, output=True):
	s = game.startState()

	a1_turn = True

	winner = game.gameOver(s)

	while not winner:
		if output:
			game.printState(s)
			print('\n**************')
		if a1_turn:
			a = a1.getMove(s)
			a1_turn = False
		else:
			a = a2.getMove(s)
			a1_turn = True
		if a == None:
			break
		s = game.nextState(s, a)
		winner = game.gameOver(s)
	
	if output:
		game.printState(s)
		print("Winner: {}".format(winner))

	return winner

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
