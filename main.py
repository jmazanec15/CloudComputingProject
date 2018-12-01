#!/usr/bin/env cctools_python
# CCTOOLS_PYTHON_VERSION 2.7 2.6

import numpy as np
import subprocess
import time

from work_queue import *
from games.connect4 import Connect4
from nn import NN
from agent import ComputerAgent, HumanAgent

from work_queue import *

def workers(games_per_task, games_per_iter):
	port = 9139

	keras_path = "/afs/crc.nd.edu/user/l/lwurl/.local/lib/python2.7/site-packages/keras"
	h5py_path = "/afs/crc.nd.edu/user/l/lwurl/.local/lib/python2.7/site-packages/h5py"
	keras_applications_path = "/afs/crc.nd.edu/user/l/lwurl/.local/lib/python2.7/site-packages/keras_applications"
	keras_preprocessing_path = "/afs/crc.nd.edu/user/l/lwurl/.local/lib/python2.7/site-packages/keras_preprocessing"
	yaml_path = "/afs/crc.nd.edu/user/l/lwurl/.local/lib/python2.7/site-packages/yaml"
	numpy_path = "/afs/crc.nd.edu/x86_64_linux/t/tensorflow/1.6/gcc/python2/build/lib/python2.7/site-packages/numpy"
	script_path = "/afs/crc.nd.edu/user/l/lwurl/Cloud/script.sh"
	cloud_path = "/afs/crc.nd.edu/user/l/lwurl/Cloud"

	try:
		q = WorkQueue(port)
	except:
		sys.exit(1)

	print "listening on port %d..." % q.port

	for i in range(games_per_iter/games_per_task):
		train_file = 'training_examples' + str(i) + '.npy'
		policies_file = 'policies' + str(i) + '.npy'
		values_file = 'values' + str(i) + '.npy'
		outfile = "errors.txt"

		command = "./script.sh " + str(games_per_task) + " " + str(i) + " >> %s 2>&1" % (outfile)

		t = Task(command)

		t.specify_file(keras_path, "keras", WORK_QUEUE_INPUT, cache=True)
		t.specify_file(h5py_path, "h5py", WORK_QUEUE_INPUT, cache=True)
		t.specify_file(cloud_path, "cloud", WORK_QUEUE_INPUT, cache=True)
		t.specify_file(keras_applications_path, "keras_applications", WORK_QUEUE_INPUT, cache=True)
		t.specify_file(keras_preprocessing_path, "keras_preprocessing", WORK_QUEUE_INPUT, cache=True)
		t.specify_file(yaml_path, "yaml", WORK_QUEUE_INPUT, cache=True)
		t.specify_file(numpy_path, "numpy", WORK_QUEUE_INPUT, cache=True)
		t.specify_file(script_path, "script.sh", WORK_QUEUE_INPUT, cache=True)
		
		t.specify_file(train_file, train_file, WORK_QUEUE_OUTPUT, cache=False)
		t.specify_file(policies_file, policies_file, WORK_QUEUE_OUTPUT, cache=False)
		t.specify_file(values_file, values_file, WORK_QUEUE_OUTPUT, cache=False)
		t.specify_file(outfile, outfile, WORK_QUEUE_OUTPUT, cache=False)

		# Once all files has been specified, we are ready to submit the task to the queue.
		taskid = q.submit(t)
		print "submitted task (id# %d): %s" % (taskid, t.command)

	print "waiting for tasks to complete..."
	while not q.empty():
		t = q.wait(5)
		if t:
			print "task (id# %d) complete: %s (return code %d)" % (t.id, t.command, t.return_status)
			if t.return_status != 0:
				i = t.id - 1
				train_file = 'training_examples' + str(i) + '.npy'
				policies_file = 'policies' + str(i) + '.npy'
				values_file = 'values' + str(i) + '.npy'
				outfile = "errors.txt"

				command = "./script.sh " + str(games_per_task) + " " + str(i) + " >> %s 2>&1" % (outfile)

				t = Task(command)

				t.specify_file(keras_path, "keras", WORK_QUEUE_INPUT, cache=True)
				t.specify_file(h5py_path, "h5py", WORK_QUEUE_INPUT, cache=True)
				t.specify_file(cloud_path, "cloud", WORK_QUEUE_INPUT, cache=True)
				t.specify_file(keras_applications_path, "keras_applications", WORK_QUEUE_INPUT, cache=True)
				t.specify_file(keras_preprocessing_path, "keras_preprocessing", WORK_QUEUE_INPUT, cache=True)
				t.specify_file(yaml_path, "yaml", WORK_QUEUE_INPUT, cache=True)
				t.specify_file(numpy_path, "numpy", WORK_QUEUE_INPUT, cache=True)
				t.specify_file(script_path, "script.sh", WORK_QUEUE_INPUT, cache=True)
				
				t.specify_file(train_file, train_file, WORK_QUEUE_OUTPUT, cache=False)
				t.specify_file(policies_file, policies_file, WORK_QUEUE_OUTPUT, cache=False)
				t.specify_file(values_file, values_file, WORK_QUEUE_OUTPUT, cache=False)
				t.specify_file(outfile, outfile, WORK_QUEUE_OUTPUT, cache=False)

				# Once all files has been specified, we are ready to submit the task to the queue.
				taskid = q.submit(t)
				print "REsubmitted task (id# %d): %s" % (taskid, t.command)
				#task object will be garbage collected by Python automatically when it goes out of scope

	print "all tasks complete!"

	#work queue object will be garbage collected by Python automatically when it goes out of scope
	
	return "ok"

def main():
	'''
		This function should generate a trained model that will allow us
		to play the games
	'''
	### Parameters
	input_shape = (6, 7, 1)
	iterations = 10
	games_per_iter = 1000	# 100
	head_to_head_games = 5	# 50
	threshold = 0.55
	cpuct = 1
	epochs = 2	# 10
	batch_size = 64
	training_size = 10000

	### Setup variables
	nn = NN(input_shape)
	game = Connect4()
	training_examples = np.array([])
	policies = list()
	values = list()

	games_per_task = 5

	f = open("{}_games_per_iter_100workers_{}_games_per_task.csv".format(games_per_iter, games_per_task), "a")

	for it in range(iterations):
		print("ITERATION: {}/{}".format(it+1, iterations))
		
		### Self playing to accumulate training examples
		print("Self playing {} games".format(games_per_iter))

		ti = time.time()

		workers(games_per_task, games_per_iter)
		for i in range(games_per_iter/games_per_task):
			train_file = 'training_examples' + str(i) + '.npy'
			policies_file = 'policies' + str(i) + '.npy'
			values_file = 'values' + str(i) + '.npy'
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

		tf = time.time()

		it_time = tf-ti
		f.write(str(it_time))
		f.write('\n')

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

	playGameHumanVsComp(game, nn, cpuct)

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
		print('')
		t = game.gameOver(s)
		if t:
			print("{} wins".format(t))
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
