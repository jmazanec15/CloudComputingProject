'''
	Monte Carlo Tree Search class
	This class is used to execute the MCTS when searching
	for the next move to play
'''

class State(object):
	def __init__(self, s):
		self.state = s
		self.actions = [] # The prior probability of selecting action a

	def isLeaf():
		if len(actions) == 0:
			return True
		return False

class Action(object):
	def __init__(self, initState, resultState, a, prob):
		self.initState = initState
		self.resultState = resultState
		self.action = a

		self.vals = {'N':0,'W':0,'Q':0,'P':prob}

class MCTS(object):
	def __init__(self, game, root, cpuct):
		self.game = game
		self.root = root
		
		self.tree = dict()
		self.addState(root)

		self.cpuct = cpuct

    def downToLeaf(self):
        trail = []
        currentNode = self.root

        while not currentNode.isLeaf():
            ''' Choose the path based on explored / favorable '''

            maxProb = -1
            actionToTakeIndex = -1
            for index, action in enumerate(currentNode.actions):
                if action.vals['P'] > maxProb:
                    actionToTakeIndex = index
            
            result, value = currentNode.state.takeAction(currentNode.actions[actionToTakeIndex])
            currentNode = currentNode.actions[actionToTakeIndex]
            trail.append(currentNode.actions[actionToTakeIndex])

        
        return currentNode, trail, value
                    

    def backProp(self, leaf, value, trail):

		currentPlayer = leaf.state.playerTurn

		for action in trail:
			playerTurn = action.playerTurn
			if playerTurn == currentPlayer:
				direction = 1
			else:
				direction = -1

			action.stats['N'] = action.stats['N'] + 1
			action.stats['W'] = action.stats['W'] + value * direction
			action.stats['Q'] = action.stats['W'] / action.stats['N']

	def addState(self, state):
		self.tree[self.game.stateToId(state)] = state
