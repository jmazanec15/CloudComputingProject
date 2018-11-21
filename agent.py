from mcts import MCTS

class Agent(object):
	'''
		Agent class that plays the game
	'''
	def __init__(self, game):
		self.game = game

	def getMove(self, s):
		''' 
			Abstract method, must return a valid action
		'''
		pass


class ComputerAgent(Agent):
	'''
		Agent uses neural net to play
	'''
	def __init__(self, game, cpuct, nn):
		super(ComputerAgent, self).__init__(game)
		self.mcts = MCTS(game, game.startState(), cpuct, nn)


	def getMove(self, s):
		valid_moves = self.game.getValidActions(s)
		a = self.mcts.getMove(state=s)
		print(valid_moves)
		print(a)
		if a in valid_moves:
			return a

		return None


class HumanAgent(Agent):
	'''
		Agent uses human to play
	'''
	def __init__(self, game):
		super(HumanAgent, self).__init__(game)

	def getMove(self, s):
		valid_moves = self.game.getValidActions(s)
		print(valid_moves)
		while 1:
			try:
				a = raw_input('Make a move: ')
				if a == 'q':
					return None
				a = int(a)
			except:
				continue

			if a in valid_moves:
				return a
			return None
		