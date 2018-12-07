import numpy as np
from params import *
'''
	Monte Carlo Tree Search class
	This class is used to execute the MCTS when searching
	for the next move to play
'''

class Node(object):
	def __init__(self, s):
		self.state = s
		self.edges = list()
		self.v = None


class Edge(object):
	def __init__(self, in_state, out_state, a, prob):
		self.in_state = in_state
		self.out_state = out_state
		self.action = a
		self.vals = {'N':0,'W':0,'Q':0,'P':prob}

class MCTS(object):
	def __init__(self, game, start_state, cpuct, nn):
		self.game = game
		self.nn = nn
		self.cpuct = cpuct

		self.tree = dict()

		self.root = Node(start_state)
		self._addNode(self.root)
		self.root.v = self._initEdges(self.root)


	def getMove(self, iterations=10, state=[], get_policy=False):
		'''
			For a certain state, get the best potential move
	
			If state is passed in, set it as the root of the tree
			Otherwise, just use the current root of the tree
		'''
		if len(state) != 0:
			str_rep = self.game.stateToId(state)
		 	if str_rep in self.tree:
				self.root = self.tree[str_rep]
			else:
				self.root = Node(state)
				self._addNode(self.root)
				self.root.v = self._initEdges(self.root)

		## Run search and update on the tree
		for i in range(iterations):
			leaf_node, trail, max_edge = self._search()
			self._update(leaf_node, trail, max_edge)


		## Get the best action from the tree policy - no neural net used
		max_pi = -float('inf')
		max_edge = None

		Nall = float(sum([e.vals['N'] for e in self.root.edges]))

		policies = np.zeros(42)

		for edge in self.root.edges:
			pi = edge.vals['N'] / (Nall+1)

			policies[edge.action] = pi

			if pi > max_pi:
				max_pi = pi
				max_edge = edge		

		if get_policy:
			return max_edge.action, policies

		return max_edge.action

	def _search(self):
		''' 
			Find the leaf along the path that maximizes U and the max_edge
	
			return that node, the trail to get there and whether that
			node is a terminal state
    	'''
		trail = list()
		str_rep = self.game.stateToId(self.root.state)
		root_rep = str_rep
		## While the node is in the tree, continue
		# This loop will always be entered
		while str_rep in self.tree:
			curr_node = self.tree[str_rep]
			if str_rep == root_rep:
				max_edge = self._maxEdge(curr_node, randomness=True)
			else:
				max_edge = self._maxEdge(curr_node)

			# If max_edge is None, the game must be over
			if not max_edge:
				return curr_node, trail, None

			trail.append(max_edge)
			str_rep = self.game.stateToId(max_edge.out_state)

		return curr_node, trail, max_edge


	def _update(self, leaf_node, trail, max_edge):
		'''
			Given a leaf node, update each edge in trail

			If terminal, use that value as the v needed to update the edges
			Otherwise, create a new node, and use the NN to get v and p
		'''
		if not max_edge:
			v = leaf_node.v
		else:
			new_node = Node(max_edge.out_state)
			self._addNode(new_node)
			new_node.v = self._initEdges(new_node)
			v = new_node.v

		## Back fill v through the trail
		for edge in trail:
			edge.vals['N'] += 1
			edge.vals['W'] += v * (edge.in_state[-1] * leaf_node.state[-1])
			edge.vals['Q'] = edge.vals['W'] / edge.vals['N']


	def _initEdges(self, node):
		'''
			Initialize all edges for a given node and return the 
			value predicted by the neural network for that node
		'''
		actions = self.game.getValidActions(node.state)
		data = np.array([self.game.convertStateForNN(node.state)])
		p, v = self.nn.predict(data)

		for a in actions:
			outState = self.game.nextState(node.state, a)
			node.edges.append(Edge(node.state, outState, a, p[0][a]))

		return v[0]


	def _maxEdge(self, node, randomness=None):
		''' 
			Return the edge that maximizes UCB 
		
			U(s, a) = Q(s, a) + cpuct * P(s, a) * sqrt((sum of all N)/(1 + N(s,a)))s
		'''

		# Add randomness here so that an NN playing against it self doesnt generate the 
		# same states
		if randomness:
			epsilon = EPSILON
			nu = np.random.dirichlet([ALPHA] * len(node.edges)) 
		else:
			epsilon = 0
			nu = [0] * len(node.edges)

		max_edge = None
		max_U = -float('inf')
		Nall = sum([e.vals['N'] for e in node.edges])
		for i, edge in enumerate(node.edges):
			
			U = edge.vals['Q'] + self.cpuct * ((1-epsilon) * edge.vals['P'] + epsilon * nu[i])* np.sqrt(Nall)/(1+edge.vals['N'])

			if U > max_U:
				max_U = U
				max_edge = edge

		return max_edge


	def _addNode(self, node):
		''' 
			Add node to the tree for later lookup 
		'''
		self.tree[self.game.stateToId(node.state)] = node
