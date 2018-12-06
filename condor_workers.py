from work_queue import *
from params import *

''' 
	File contains code for submitting workers to play games to condor
'''

def submit_task(q, i_num, games_per_task, resub=False):
	train_file = 'training_examples{}.npy'.format(i_num)
	policies_file = 'policies{}.npy'.format(i_num)
	values_file = 'values{}.npy'.format(i_num)
	outfile = 'errors.txt'

	command = './script.sh {} {} >> {} 2>&1'.format(games_per_task, i_num, outfile)

	keras_path = '/afs/crc.nd.edu/user/{}/{}/.local/lib/python2.7/site-packages/keras'.format(USERNAME[0], USERNAME)
	h5py_path = '/afs/crc.nd.edu/user/{}/{}/.local/lib/python2.7/site-packages/h5py'.format(USERNAME[0], USERNAME)
	keras_applications_path = '/afs/crc.nd.edu/user/{}/{}/.local/lib/python2.7/site-packages/keras_applications'.format(USERNAME[0], USERNAME)
	keras_preprocessing_path = '/afs/crc.nd.edu/user/{}/{}/.local/lib/python2.7/site-packages/keras_preprocessing'.format(USERNAME[0], USERNAME)
	yaml_path = '/usr/lib64/python2.7/site-packages/yaml'
	numpy_path = '/afs/crc.nd.edu/user/{}/{}/.local/lib/python2.7/site-packages'.format(USERNAME[0], USERNAME)
	script_path = '/afs/crc.nd.edu/user/{}/{}/CloudComputingProject/script.sh'.format(USERNAME[0], USERNAME)
	cloud_path = '/afs/crc.nd.edu/user/{}/{}/CloudComputingProject'.format(USERNAME[0], USERNAME)

	t = Task(command)

	t.specify_file(keras_path, 'keras', WORK_QUEUE_INPUT, cache=True)
	t.specify_file(h5py_path, 'h5py', WORK_QUEUE_INPUT, cache=True)
	t.specify_file(cloud_path, 'cloud', WORK_QUEUE_INPUT, cache=True)
	t.specify_file(keras_applications_path, 'keras_applications', WORK_QUEUE_INPUT, cache=True)
	t.specify_file(keras_preprocessing_path, 'keras_preprocessing', WORK_QUEUE_INPUT, cache=True)
	t.specify_file(yaml_path, 'yaml', WORK_QUEUE_INPUT, cache=True)
	t.specify_file(numpy_path, 'numpy', WORK_QUEUE_INPUT, cache=True)
	t.specify_file(script_path, 'script.sh', WORK_QUEUE_INPUT, cache=True)
	
	t.specify_file('game_data/{}'.format(train_file), train_file, WORK_QUEUE_OUTPUT, cache=False)
	t.specify_file('game_data/{}'.format(policies_file), policies_file, WORK_QUEUE_OUTPUT, cache=False)
	t.specify_file('game_data/{}'.format(values_file), values_file, WORK_QUEUE_OUTPUT, cache=False)
	t.specify_file(outfile, outfile, WORK_QUEUE_OUTPUT, cache=False)

	taskid = q.submit(t)
	if not resub:
		print('Submitted task (id# {}): {}'.format(taskid, t.command))
	else:
		print('Re-submitted task (id# {}): {}'.format(taskid, t.command))
	return taskid


def workers(games_per_iter, games_per_task):
	try:
		q = WorkQueue(PORT)
	except:
		sys.exit(1)

	print('listening on port {}...'.format(q.port))

	for i in range(games_per_iter/games_per_task):
		task_id = submit_task(q, i, games_per_task)


	print('Waiting for tasks to complete...')
	while not q.empty():
		t = q.wait(5)
		if t:
			print('task (id# {}) complete: {} (return code {})'.format(t.id, t.command, t.return_status))
			if t.return_status != 0:
				task_id = submit_task(q, games_per_task, t.id - 1, resub=True)

	print('Tasks complete!')
	return 0
