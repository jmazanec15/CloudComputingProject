'''
	Implemenation of Connect4 game for
	AlphaGo Zero Project
	10/12/2018
'''

import copy

class Connect4(object):
	def __init__(self):
		self.turn = 1 # 1 for black -1 for white
		self.size = 42
		self.winStates = [
						  # Horizontal - 28
						  [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6],
						  [7,8,9,10],[8,9,10,11],[9,10,11,12],[10,11,12,13],
						  [14,15,16,17],[15,16,17,18],[16,17,18,19],[17,18,19,20],
						  [21,22,23,24],[22,23,24,25],[23,24,25,26],[24,25,26,27],
						  [28,29,30,31],[29,30,31,32],[30,31,32,33],[31,32,33,34],
						  [35,36,37,38],[36,37,38,39],[37,38,39,40],[38,39,40,41],
						  # Vertical - 21
						  [0,7,14,21],[7,14,21,28],[14,21,28,35],
						  [1,8,15,22],[8,15,22,29],[15,22,29,36],
						  [2,9,16,23],[9,16,23,30],[16,23,30,37],
						  [3,10,17,24],[10,17,24,31],[17,24,31,38],
						  [4,11,18,25],[11,18,25,32],[18,25,32,39],
						  [5,12,19,26],[12,19,26,33],[19,26,33,40],
						  [6,13,20,27],[13,20,27,34],[20,27,34,41],
						  #up, right - 12
						  [21,15,9,3],[28,22,16,10],[35,29,23,17],
						  [22,16,10,4],[29,23,17,11],[36,30,24,18],
						  [23,17,11,5],[30,24,18,12],[37,31,25,19],
						  [24,18,12,6],[31,25,19,13],[38,32,26,20],
						  #down, right - 12
						  [0,8,16,24],[7,15,23,31],[14,22,30,38],
						  [1,9,17,25],[8,16,24,32],[15,23,31,39],
						  [2,10,18,26],[9,17,25,33],[16,24,32,40],
						  [3,11,19,27],[10,18,26,34],[17,25,33,41]
						]

	def startState(self):
		'''
			State should be an array of 43 vals where the 
			first 42 are the squares and the last is the turn

			Board looks like:
			 0  1  2  3  4  5  6
			 7  8  9 10 11 12 13
			14 15 16 17 18 19 20
			21 22 23 24 25 26 27
			28 29 30 31 32 33 34
			35 36 37 38 39 40 41

		'''
		state = [0 for _ in range(self.size + 1)]
		state[self.size] = 1 # set turn to 1: black
		return state

	def getValidActions(self, s):
		actions = list()
		rows = 6
		cols = 7
		for c in range(cols):
			base = 35 + c
			for r in range(rows):
				cell = base-r*7
				if s[cell] == 0:
					actions.append(cell)
					break
		return actions

	def nextState(self, s, a):
		n = copy.copy(s)
		
		if n[a] == 0:
			n[a] = n[-1]
			n[-1] *= -1
			return n

		return None

	def gameOver(self, s):
		# Need to check if someone won or if the board is full
		l = s[-1]*-1 # Whoever made the last move
		for w in self.winStates:
			if s[w[0]]*l + s[w[1]]*l + s[w[2]]*l + s[w[3]]*l == 4:
				return l

		for i in range(42):
			if s[i] == 0:
				return 0

		return -l

	def printState(self, s):
		t = {1: 'B', 0: '0', -1: 'W'}
		for r in range(6):
			for c in range(7):
				print t[s[r*7+c]],
			print('')
