import numpy as np

'''
	Monte Carlo Tree Search class
	This class is used to execute the MCTS when searching
	for the next move to play
'''

class Node(object):
	def __init__(self, s):
		self.state = s
		self.turn = s[-1] # last value in the state is the turn of the player
		self.edges = list()
		self.v = None

	def isLeaf(self):
		if len(self.edges) == 0:
			return True
		return False

class Edge(object):
	def __init__(self, in_state, out_state, a, prob):
		self.in_state = in_state
		self.out_state = out_state
		self.action = a
		self.turn = in_state.turn
		self.vals = {'N':0,'W':0,'Q':0,'P':prob}

class MCTS(object):
	def __init__(self, game, start_state, cpuct, nn):
		self.game = game
		self.nn = nn
		self.end_states = dict()
		self.cpuct = cpuct

		self.root = Node(start_state)
		self.tree = dict()
		self.addNode(self.root)
		self.root.v = self.initEdges(self.root)


	def getMove(self, iterations=10, state=None):
		'''
			For a certain state, get the best potential move
		'''
		str_rep = self.game.stateToId(state)
		
		if state and self.tree[str_rep]:
			self.root = self.tree[str_rep]
		elif state:
			self.root = Node(state)
			self.addNode(self.root)
			self.root.v = self.initEdges(self.root)
		else:
			pass

		## Run search and update on the tree
		for i in range(iterations):
			leaf_node, trail, terminal = self.search()
			self.update(leaf_node, trail, terminal)


		## Get the best action from the tree policy
		max_pi = -float('inf')
		max_edge = None

		Nall = float(sum([e.vals['N'] for e in self.root.edges]))

		for edge in self.root.edges:
			pi = edge.vals['N'] / (Nall+1)

			if pi > max_pi:
				max_pi = pi
				max_edge = edge		

		return max_edge.a

	def search(self):
		''' 
			Find the leaf along the path that maximizes U
	
			return that node, the trail to get there and whether that
			node is a terminal state
    	'''
		trail = list()
		curr_node = self.root
		str_rep = self.game.stateToId(curr_node.state)

		while not curr_node.isLeaf():
			max_edge = self._maxEdge(curr_node)
			trail.append(max_edge)
			str_rep = self.game.stateTostr(max_edge.out_state)
			curr_node = self.tree[str_rep] # go to next node

		## Bookkeeping about endstates
		if str_rep in self.end_states:
			terminal = self.end_states[str_rep]
		else:
			terminal = self.game.gameOver(curr_node.state)
			self.end_states[str_rep] = terminal

		return curr_node, trail, terminal


	def update(self, leaf_node, trail, terminal):
		'''
			Given a leaf node, update each edge in trail

			If terminal, use that value as the v needed to update the edges
			Otherwise, create a new node, and use the NN to get v and p
		'''
		if terminal:
			v = terminal
		else:
			max_edge = self._maxEdge(leaf_node)
			new_node = Node(max_edge.out_state)

		## Back fill v through the trail
		for edge in trail:
			sign = edge.turn * leaf.turn
			edge.vals['N'] += 1
			edge.vals['W'] += v * sign
			edge.vals['Q'] = edge.vals['W'] / edge.vals['N']


	def initEdges(self, node):
		'''
			Initialize all edges for a given node and return the 
			value predicted by the neural network for that node
		'''
		actions = self.game.getValidActions(node.state)
		
		v, p = self.nn.predict(node.state)

		for a in actions:
			outState = self.game.nextState(node.state, a)
			node.edges.append(Edge(node.state, outState, a, p[a]))

		return v


	def _maxEdge(self, node, randomness=None):
		''' 
			Return the edge that maximizes UCB 
		
			U(s, a) = Q(s, a) + cpuct * P(s, a) * sqrt((sum of all N)/(1 + N(s,a)))s
		'''
		# TODO: add optional randomness
		max_edge = None
		max_U = -float('inf')
		Nall = sum([e.vals['N'] for e in node.edges])
		for edge in node.edges:
			U = edge.vals['Q'] + self.cpuct * edge.vals['P'] * np.sqrt(Nall)/(1+edge.vals['N'])

			if U > max_U:
				max_U = U
				max_edge = edge

		return max_edge


	def addNode(self, node):
		''' 
			Add node to the tree for later lookup 
		'''
		self.tree[self.game.stateToId(node.state)] = node

