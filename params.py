'''
	Set the values of the neural network here
'''

params = dict()

params['reg_const'] = 0.0001
params['learning_rate'] = .1
params['momentum'] = 0.9

## For randomness
params['epsilon'] = 0.2
params['alpha'] = 0.8

params['base_layers'] = [
							{'filters': 75, 'kernel_size': 4},
							{'filters': 75, 'kernel_size': 4},
							{'filters': 75, 'kernel_size': 4}
						]


## Should have epochs, batch size, ...
# input_shape = (6, 7)
# iterations = 10
# games_per_iter = 25
# head_to_head_games = 100
# threshold = 0.55
# cpuct = 1
# epochs = 10
# batch_size = 64
