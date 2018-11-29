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
	nn = NN((6,7,1), path_to_nn="./models/v1.h5", load=True)
	game = Connect4()
	playGameHumanVsComp(game, nn, 1)	


def playGameHumanVsComp(game, nn, cpuct, first=True):
	if first:
		a1 = ComputerAgent(game, cpuct, nn)
		a2 = HumanAgent(game)
	else:
		a1 = HumanAgent(game)
		a2 = ComputerAgent(game, cpuct, nn)

	return playGame(game, a1, a2)

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


if __name__ == "__main__":
	main()
