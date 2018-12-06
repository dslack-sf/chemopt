import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

"""
 * Want to get center of data set
 * Need to start the recommender here
"""

state_space = pd.read_csv("EtNH3Istateset.csv")
center = []
names = []

NUM_INORGANIC_STEPS = 4
NUM_ORGANIC_STEPS = 4
NUM_ACIDIC_STEPS = 3
TOTAL_ALLOWED_REACTIONS = 48

assert(TOTAL_ALLOWED_REACTIONS == NUM_INORGANIC_STEPS * NUM_ORGANIC_STEPS * NUM_ACIDIC_STEPS)

_rxn_M_inorganic_max = max(state_space['_rxn_M_inorganic'])
_rxn_M_inorganic_min = min(state_space['_rxn_M_inorganic'])

_rxn_M_organic_max = max(state_space['_rxn_M_organic'])
_rxn_M_organic_min = min(state_space['_rxn_M_organic'])

_rxn_M_acid_max = max(state_space['_rxn_M_acid'])
_rxn_M_acid_min = min(state_space['_rxn_M_acid'])

step_rxn_M_inroganix = (_rxn_M_inorganic_max - _rxn_M_inorganic_min) / NUM_INORGANIC_STEPS
step_rxn_M_organic = (_rxn_M_organic_max - _rxn_M_organic_min) / NUM_ORGANIC_STEPS
step_rxn_M_acid = (_rxn_M_acid_max - _rxn_M_acid_min) / NUM_ACIDIC_STEPS

#Lay out reaction space:
result = []
cur_inorganic = _rxn_M_inorganic_min
cur_acid = _rxn_M_acid_min
cur_organic = _rxn_M_organic_min
for _ in range(NUM_INORGANIC_STEPS):
	cur_inorganic += step_rxn_M_inroganix
	for _ in range(NUM_ORGANIC_STEPS):
		cur_organic += step_rxn_M_organic
		for _ in range(NUM_ACIDIC_STEPS):
			cur_acid += step_rxn_M_acid
			result.append([cur_inorganic, cur_organic, cur_acid])
		cur_acid = _rxn_M_acid_min
	cur_organic = _rxn_M_organic_min


target_df = pd.DataFrame(result, columns = ['_rxn_M_inorganic', '_rxn_M_organic', '_rxn_M_acid'])
ids = []

for i, row in target_df.iterrows():
	distances = euclidean_distances(state_space, [row])
	distances = list(distances)
	min_distance = min(distances)
	indx = distances.index(min_distance)
	ids.append(indx)

results = pd.DataFrame(ids, columns = ['desired_ids'])
results.to_csv('starting_ids.csv')








