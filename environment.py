import numpy as np

### ENVIRONMENT ###

class Environment:
	'''The class that represents our pretty environment'''
	default_parameters = {
		'width' : 10,
		'height': 10,
		'obstacles': [ (2,2), (2,3), (2,4), (6,7), (7,7) ],
		'nb_trashes': 30
	}

	def __init__(self, w=0, h=0, nb_trashes=0):
		# width of the environement (not including walls)
		self.width = self.default_parameters['width'] if w == 0 else w

		# height of the environment (not including walls)
		self.height = self.default_parameters['height'] if h == 0 else h

		# number of trashes in the environment
		self.nb_trashes = self.default_parameters['nb_trashes'] if nb_trashes == 0 else nb_trashes

		# set positions of obstacles
		self.obstacles = self.default_parameters['obstacles']

		# randomize positions of trashes
		self.trashes = []
		i = 0
		while i < self.nb_trashes:
			x = np.random.randint(0, self.width)
			y = np.random.randint(0, self.height)
			if (x,y) not in self.trashes and (x,y) not in self.obstacles:
				self.trashes.append((x,y))
				i += 1



	def display(self):
		for i in range(self.height+1):
			for j in range(self.width+1):
				if j < self.width:
					if (i,j) in self.trashes:
						symbol = '*'
					elif (i,j) in self.obstacles:
						symbol = '#'
					else:
						symbol = ' '
					print('| %s ' % symbol, end='', flush=True) # don't bother these parameters I only use those to print on same line
				else:
					print('|', end='', flush=True)
			print()