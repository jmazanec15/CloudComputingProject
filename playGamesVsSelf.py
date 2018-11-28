from nn import NN
from agent import ComputerAgent, HumanAgent
from games.connect4 import Connect4
import sys
import numpy as np

if __name__ == '__main__':
    pathToNN = sys.argv[1]
    numGames = int(sys.argv[2])

    # Load nn form path
    nn = NN(None, pathToNN, load=True)
    cpuct = 1

    a1 = ComputerAgent(game, cpuct, nn)
    a2 = ComputerAgent(game, cpuct, nn)
    game = Connect4()

    # Play the games
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