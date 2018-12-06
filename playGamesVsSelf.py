import numpy as np
import sys

from nn import NN
from agent import ComputerAgent
from games.connect4 import Connect4
from params import *

def main():
	'''
		This short python file should play a certain number of games against itself
		and write the resulting numpy arrays to files

		The first argument is the path to the neural network
		The second argument is how many games it should play and the third is
		for keeping track of which worker this is for the output files
	'''
	path_to_nn = sys.argv[1]
	num_games = int(sys.argv[2])
	worker_num = sys.argv[3]

	# Load nn form path
	# Currently not working on condor so we are just creating a nn from scratch
	#nn = NN(INPUT_SHAPE, path_to_nn, load=True)
	nn = NN(INPUT_SHAPE)
	game = Connect4()

	# Lists to keep track of data generated
	training_examples = np.array([])
	policies = list()
	values = list()

	# Loop through and play a certain number of games
	for i in range(num_games):
		a1 = ComputerAgent(game, CPUCT, nn)
		a2 = ComputerAgent(game, CPUCT, nn)

		s = game.startState()

		examples = list()
		ps = list()
		vs = list()

		a1_turn = True
		winner = game.gameOver(s)

		# Playing actual game
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

		if len(training_examples) != 0:
			training_examples = np.append(training_examples, examples, axis=0)
			policies = np.append(policies, ps, axis=0)
			values = np.append(values, vs, axis=0)
		else:
			training_examples = examples
			policies = ps
			values = vs

	training_examples = np.array(training_examples)
	policies = np.array(policies)
	values = np.array(values)

	# Save the training examples
	np.save('training_examples{}'.format(worker_num), training_examples)
	np.save('policies{}'.format(worker_num), policies)
	np.save('values{}'.format(worker_num), values)


if __name__ == '__main__':
    main()
    sys.exit(0)
