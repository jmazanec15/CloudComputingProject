'''
	Implemenation of Connect4 game for
	AlphaGo Zero Project
	10/12/2018
'''

import copy

class Connect4(object):
	def __init__(self):
		pass

	def startState(self):
		return [[0 for _ in range(self.columns)] for _ in range(self.rows)]

	def getActions(self, s):
		actions = list()
		for c in range(self.columns):
			if s[self.rows-1][c] == 0:
				actions.append(c)
		return actions

	def nextState(self, s, a, t):
		n = copy.copy(s)
		for i in range(self.rows):
			if n[i][a] == 0:
				n[i][a] = t
				return n

		return None

	def gameOver(self, s):
		full = -1

		for r in range(self.rows):
			for c in range(self.columns):
				if s[r][c] == 0: 
					full = 0
					continue
				if r < 3 and s[r][c] == s[r][c+1] and s[r][c] == s[r][c+2] and s[r][c] == s[r][c+3]:
					return 1
				elif r < 3 and c < 4 and s[r][c] == s[r+1][c+1] and s[r][c] == s[r+2][c+2] and s[r][c] == s[r+3][c+3]:
					return 1
				elif c < 4 and s[r][c] == s[r][c+1] and s[r][c] == s[r][c+2] and s[r][c] == s[r][c+3]:
					return 1

		return full
