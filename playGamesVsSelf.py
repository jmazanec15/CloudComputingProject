import boto3
from nn import NN
from agent import ComputerAgent, HumanAgent
from connect4 import Connect4
import sys
import numpy as np

if __name__ == '__main__':
    pathToNN = sys.argv[1]
    numGames = int(sys.argv[2])
    workerNum = sys.argv[3]

    # Load nn form path
    nn = NN((6, 7, 1), pathToNN, load=True)
    cpuct = 1

    game = Connect4()

    training_examples = np.array([])
    policies = list()
    values = list()

    for i in range(numGames):
        a1 = ComputerAgent(game, cpuct, nn)
        a2 = ComputerAgent(game, cpuct, nn)

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

        if len(training_examples) != 0:
            training_examples = np.append(
                training_examples, examples, axis=0)
            policies = np.append(policies, ps, axis=0)
            values = np.append(values, vs, axis=0)
        else:
            training_examples = examples
            policies = ps
            values = vs

    training_examples = np.array(training_examples)
    policies = np.array(policies)
    values = np.array(values)

    np.save("training_examples" + workerNum, training_examples)
    np.save("policies" + workerNum, policies)
    np.save("values" + workerNum, values)

    # Upload files to S3
    bucket_name = 'cloud-computing-alpha-zero-bucket'
    s3 = boto3.client('s3')
    s3.upload_file('training_examples' + workerNum + '.npy', bucket_name, 'training_examples_' + workerNum)
    s3.upload_file('policies' + workerNum + '.npy', bucket_name, 'policies_' + workerNum)
    s3.upload_file('values' + workerNum + '.npy', bucket_name, 'values_' + workerNum)

    sys.exit(0)