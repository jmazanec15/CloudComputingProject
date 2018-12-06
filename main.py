#!/usr/bin/env cctools_python
# CCTOOLS_PYTHON_VERSION 2.7 2.6

import numpy as np
import subprocess
import sys
import time
if len(sys.argv) > 1 and sys.argv[1] == 'condor':
	from condor_workers import workers

from agent import ComputerAgent
from games.connect4 import Connect4
from nn import NN
from params import *

def main():
	'''
		This function should generate a trained model that will allow us
		to play the games
	'''
	## specifiy condor to train on condor
	use_condor = True if len(sys.argv) > 1 and sys.argv[1] == 'condor' else False

	## Setup variables
	curr_nn = NN(INPUT_SHAPE)
	best_nn = curr_nn
	game = Connect4()
	training_examples = np.array([])
	policies = np.array([])
	values = np.array([])

	## Time per X games played in 1 iteration
	if use_condor:
		result_file = open('results/workqueue_{}_games_per_iter_{}_workers_{}_games_per_task.csv'.format(GAMES_PER_ITERATION, WORKERS, GAMES_PER_TASK), "a")
	else:
		result_file = open('results/single_{}_games_per_iter_{}_workers_{}_games_per_task.csv'.format(GAMES_PER_ITERATION, WORKERS, GAMES_PER_TASK), "a")

	for it in range(ITERATIONS):
		print('ITERATION: {}/{}\nSelf playing {} games'.format(it+1, ITERATIONS, GAMES_PER_ITERATION))
		
		## Self play section
		ti = time.time() # Take the time to play X games
		if use_condor:
			new_examples, ps, vs = selfPlayCondor(game, best_nn, CPUCT, GAMES_PER_ITERATION, GAMES_PER_TASK)
		else:
			new_examples, ps, vs = selfPlaySingle(game, best_nn, CPUCT, GAMES_PER_ITERATION)

		tf = time.time() # Record time
		result_file.write('{}\n'.format(tf-ti))

		# Dont let np array get bigger than max_size_of_dataset
		if len(training_examples) + len(new_examples) > MAX_SIZE_OF_DATASET:
			r = len(training_examples) + len(new_examples) - MAX_SIZE_OF_DATASET
			training_examples = training_examples[r:]
			policies = policies[r:]
			values = values[r:]

		if len(training_examples) != 0:
			training_examples = np.append(training_examples, new_examples, axis=0)
			policies = np.append(policies, ps, axis=0)
			values = np.append(values, vs, axis=0)
		else:	
			training_examples = new_examples
			policies = ps
			values = vs

		## Train network
		print("Training...")
		size = min(len(training_examples), TRAINING_SIZE)
		indices = np.random.choice(range(len(training_examples)), size=size, replace=False)
		ex_subset = training_examples[indices,:]
		p_subset = policies[indices]
		v_subset = values[indices]
		curr_nn.fit(ex_subset, [p_subset, v_subset], EPOCHS, BATCH_SIZE)

		## Have the curr_net and best_net play to survive
		print("Head to head...")
		wins = 0
		for g in range(H2H_GAMES):
			if playGameNN1VsNN2(game, best_nn, curr_nn, CPUCT) > 0:
				wins+=1
		
		if wins/float(H2H_GAMES) >= NET_THRESHOLD:
			best_nn = curr_nn
			best_nn.save_model('./models/{}_games_per_iter_{}_workers_{}_games_per_task.h5'.format(GAMES_PER_ITERATION, WORKERS, GAMES_PER_TASK))


	## Save the best model
	best_nn.save_model('./models/{}_games_per_iter_{}_workers_{}_games_per_task.h5'.format(GAMES_PER_ITERATION, WORKERS, GAMES_PER_TASK))

def selfPlaySingle(game, best_nn, cpuct, games_per_iteration):
	training_examples = np.array([])
	policies = np.array([])
	values = np.array([])
	for g in range(games_per_iteration):
		new_examples, ps, vs = playGameSelfVsSelf(game, best_nn, cpuct)
		if len(training_examples) != 0:
			training_examples = np.append(training_examples, new_examples, axis=0)
			policies = np.append(policies, ps, axis=0)
			values = np.append(values, vs, axis=0)
		else:	
			training_examples = new_examples
			policies = ps
			values = vs
	return training_examples, policies, values

def selfPlayCondor(game, best_nn, cpuct, games_per_iteration, games_per_task):
	training_examples = np.array([])
	policies = np.array([])
	values = np.array([])
	workers(games_per_iteration, games_per_task)
	for i in range(games_per_iteration/games_per_task):
		# Names of files
		train_file = 'game_data/training_examples{}.npy'.format(i)
		policies_file = 'game_data/policies{}.npy'.format(i)
		values_file = 'game_data/values{}.npy'.format(i)

		# Load in the examples the workers completed
		new_examples = np.load(train_file)
		ps = np.load(policies_file)
		vs = np.load(values_file)

		if len(training_examples) != 0:
			training_examples = np.append(training_examples, new_examples, axis=0)
			policies = np.append(policies, ps, axis=0)
			values = np.append(values, vs, axis=0)
		else:	
			training_examples = new_examples
			policies = ps
			values = vs

	return training_examples, policies, values

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

	return playGame(game, a1, a2)


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

		examples.append(game.convertStateForNN(s))
		ps.append(p)

		s = game.nextState(s, a)
		winner = game.gameOver(s)


	for _ in ps:
		vs.append(winner)

	return np.array(examples), np.array(ps), np.array(vs)


def playGame(game, a1, a2):
	s = game.startState()
	a1_turn = True
	winner = game.gameOver(s)

	while not winner:
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

	return winner

if __name__ == "__main__":
	main()
