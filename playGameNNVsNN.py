#!/usr/bin/env python

import numpy as np
import sys

from agent import ComputerAgent, HumanAgent
from games.connect4 import Connect4
from nn import NN
from params import *


def main():
	'''
		This function should generate a trained model that will allow us
		to play the games
	'''
	nn1 = NN((6,7,2), path_to_nn=sys.argv[1], load=True)
	nn2 = NN((6,7,2), path_to_nn=sys.argv[2], load=True)
	game = Connect4()
	nn1_wins = 0
	#for g in range(H2H_GAMES):
	for i in range(100):
		if i%2 == 0:
			r = playGameNN1VsNN2(game, nn1, nn2, CPUCT, first=True)	
		else:
			r = -1*playGameNN1VsNN2(game, nn1, nn2, CPUCT, first=False)
		if r == 1:
			print('NN1 won game {}'.format(i+1))
			nn1_wins+=1
		else:
			print('NN2 won game {}'.format(i+1))
	print('{} won {}/{}'.format(sys.argv[1], nn1_wins, 100))
		


def playGameNN1VsNN2(game, nn1, nn2, cpuct, first=True):
	if first:
		a1 = ComputerAgent(game, cpuct, nn1)
		a2 = ComputerAgent(game, cpuct, nn2)
	else:
		a1 = ComputerAgent(game, cpuct, nn2)
		a2 = ComputerAgent(game, cpuct, nn1)

	return playGame(game, a1, a2)	


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
