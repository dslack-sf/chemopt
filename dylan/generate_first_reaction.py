import numpy as np
import pandas as pd

"""
 * Want to get center of data set
 * Need to start the recommender here
"""

state_space = pd.read_csv("../testset.csv")
state_space = state_space.drop(["RunID_vial","_rxn_organic-inchikey"], axis = 1)

center = []
names = []
center_df = pd.DataFrame()

for c in state_space:
	center_df[c] = [state_space[c].mean()]

print (center_df)

center_df.to_csv("center_Dataframe.csv")