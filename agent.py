from mcts import MCTS

class Agent(object):
	'''
		Agent class that plays the game
	'''
	def __init__(self, game):
		self.game = game

	def getMove(self, s):
		''' 
			Abstract class, must return a valid action
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
		if a not in valid_moves:
			return None
		return a


class HumanAgent(Agent):
	'''
		Agent uses human to play
	'''
	def __init__(self, game):
		super(HumanAgent, self).__init__(game)

	def getMove(self, s):
		valid_moves = self.game.getValidActions(s)
		while 1:

			try:
				a = int(raw_input('Make a move: '))
			except:
				continue	
		
			if a in valid_moves:
				return a
			print("Invalid; try again")


		