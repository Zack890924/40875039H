'''
The backward view of TD(λ) is oriented backward in time. 
At each moment we look at the current TD error and assign it backward to each prior state according to the state's eligibility trace at that time. 
The TD(λ) update reduces to the simple TD rule (TD(0)) Only the one state preceding the current one is changed by the TD error.
Can imagine that a people sit at the current state and talk back to previous state. (like figure p21) 

For example There were two behaviors, ringing the bell and turning on the lights, which eventually led to the electric shock event. 
So what exactly caused that behavior? maybe is ringing the bell, because it rings three times, 
By experence if one event is often preceded by another, then there is a good chance they are somehow related. 
But some people may say that the light caused the electric shock, because it happened just before the electric shock event. 
Therefore, we must consider these two factors at the same time when allocating credit.

for all states:
	V(st) = V(st) + alpha * delta * eligibility_trace
	eligibility_trace(st) = gamma * lambda * eligibility_trace(st)

'''
from operator import index
from matplotlib.pyplot import grid
import numpy as np
import random
np.random.seed(9527)
random.seed(9527)

class Environment():
	def __init__(self):
		self.rows = 4
		self.cols = 6 
		self.grid_world = [[  "T",  "s1",  "s2",  "s3",  "s4",  "s5"],
						   [ "s6",  "s7",  "s8",   "s9",   "W", "s10"],
						   ["s11",   "W", "s12",   "W",   "s13", "s14"],
						   ["s15", "s16", "s17", "s18", "s19", "s20"]] #T: Target, W: Wall
		self.action_to_number = {"up": 0, "right":1, "down":2, "left":3,"fly1":4,"fly6":5}
		self.action_dict = {"up": [-1,0], "right": [0, 1], "down": [1,0], "left":[0,-1],"fly1":[-2,-3],"fly6":[0,-3]}
		self.direction_dict = {0: "up", 1:"right", 2:"down", 3:"left",4:"fly1",5:"fly6"}
		self.invalid_start = ["T", "W"]

	def transfer_state(self, state_coordinates, action): #Input(state, action), Output(next state)
		current_state_coordinates = state_coordinates
		next_state_coordinates = state_coordinates + self.action_dict[action]

		if next_state_coordinates[0] < 0 or next_state_coordinates[0] > 3 or next_state_coordinates[1] < 0 or next_state_coordinates[1] > 5:# Out of board
			return current_state_coordinates
		next_state = self.grid_world[next_state_coordinates[0]][next_state_coordinates[1]]
		if next_state == "W": #Hit the wall 
			return current_state_coordinates
		return next_state_coordinates

class TD():
	def __init__(self):
		self.env = Environment()
		self.Max_iteration = 10000
		self.gamma = 0.9
		self.lam = 0.8
		self.Horizon = 10 #Max episode_length
		self.alpha=0.8

		self.values = {}
		for row in range(self.env.rows):
			for col in range(self.env.cols):
				if self.env.grid_world[row][col] not in self.env.invalid_start:
					for act in self.env.action_dict.keys():
						self.values[ ((row, col), act) ] = 0 #Initialize Q value (state, action) 

		self.returns_dict = {}
		for row in range(self.env.rows):
			for col in range(self.env.cols):
				if self.env.grid_world[row][col] not in self.env.invalid_start:
					for act in self.env.action_dict.keys():
						self.returns_dict[ ((row, col), act) ] = [0, 0] #[Mean value, Visited count]
		self.eligibility_trace = np.zeros((row+1,col+1))
	def on_new_state(self,state,newstate,reward,done,action):
		v = self.values[(tuple(state),action)]
		v_next = self.values[(tuple(newstate),action)]
		delta = reward + self.gamma * v_next - v
		self.eligibility_trace[state[0],state[1]] += 1
		

		# print(state,newstate)
		for s in np.argwhere(self.eligibility_trace != 0):
			

			
			# # s = s[0]
			if done and s == next_state:
				continue
			self.values[((s[0],s[1]),action)] += self.alpha * delta * self.eligibility_trace[(s[0],s[1])]
			
			self.eligibility_trace[(s[0],s[1])] = self.gamma * self.lam * int(self.eligibility_trace[(s[0],s[1])])
	def generate_initial_state(self):
		valid_state_list = [] 
		for state_row in range(self.env.rows):
			for state_col in range(self.env.cols):
				if self.env.grid_world[state_row][state_col] in self.env.invalid_start:
					continue
				else:   
					valid_state_list.append(np.array([state_row, state_col]))
		return valid_state_list
	def policy(self, state_coordinate): #Optimal policy, find the maximun Q(s,a) and return action  
		value = []
		valid_actions = []
		state_coordinate = np.array(state_coordinate)
		indexes = []
		s9=(1,3)
		s13=(2,4)
		s9=np.array(s9)
		s13=np.array(s13)
		if np.array_equal(state_coordinate,s9): 
			for action in self.env.action_dict.keys():
				if action == 'fly1':
					pass
				else:
					if not (state_coordinate == self.env.transfer_state(np.array(state_coordinate), action)).all():
						valid_actions.append(action)
						
		elif np.array_equal(state_coordinate,s13):
			for action in self.env.action_dict.keys():
				if action == 'fly6':
					pass
				else:
					if not (state_coordinate == self.env.transfer_state(np.array(state_coordinate), action)).all():
						valid_actions.append(action)
		else:
			for action in self.env.action_dict.keys():
				if action =='fly1' or action=='fly6':
					pass
				else:
					if not (state_coordinate == self.env.transfer_state(np.array(state_coordinate), action)).all():
						valid_actions.append(action)
		for valid_action in valid_actions:
			value.append(self.values[(tuple(state_coordinate), valid_action)])
		
		max_value = max(value)
		for valid_action in valid_actions:
			if max_value == self.values[(tuple(state_coordinate), valid_action)]:
				indexes.append(self.env.action_to_number[valid_action])
		# indexes = [index for index,x in enumerate(Q_value) if x == max_value]
		
		return self.env.direction_dict[random.choice(indexes)]

	def genetate_episode(self,state):

		done=False
		obs = state
		while not done:
	
			prev_obs = obs
			reward = -1
			action = self.policy(prev_obs)
			obs = self.env.transfer_state(prev_obs, action)
			
			if self.env.grid_world[obs[0]][obs[1]] == "T":
				reward = 100
				
				done = True
				break
			# episode.append([self.current_state_coordinates, action, reward])
			self.on_new_state(prev_obs,obs,reward,done,action)
			

	def iter(self):
		state_list = self.generate_initial_state()
		for iterration in range(100):
			

			self.current_state_coordinates = state_list[iterration%20]
			for ep in range(self.Horizon):

				self.genetate_episode(self.current_state_coordinates)




			   

	def render(self): #Show results
		output = self.env.grid_world
		for row in range(self.env.rows):
			for col in range(self.env.cols):
				if self.env.grid_world[row][col] in self.env.invalid_start:
					continue
				else:
					action = self.policy((row,col))
					output[row][col] = action

		for row in range(0, self.env.rows):
			print("-------------------------------------------------------")
			out = "| "
			for col in range(0, self.env.cols):
				out += str(output[row][col]).ljust(6) + " | "
			print(out)
		print("-------------------------------------------------------")

	

if __name__ == "__main__":

	print("************************Slides example***********************")
	TD = TD()
	initial_policy = [((0,1),"left"), ((0,2), "left"), ((0,3), "left"), ((0,4), "right"), ((0,5), "left"),
					  ((1,0), "right"), ((1,1), "right"), ((1,2), "down"), ((1,3), "up"), ((1,5), "up"),
					  ((2,0), "up"), ((2,2), "down"), ((2,4), "right"), ((2,5), "up"),
					  ((3,0), "up"), ((3,1), "left"), ((3,2),"left"), ((3,3), "right"), ((3,4), "left"), ((3,5), "left")]
	for state_action in initial_policy:
		TD.values[state_action] = 1
	print("Before (random policy), T = Target, W = Wall")
	TD.render()
	print("\n\nRunning the episodes for TD_LANB...\n\n")
	TD.iter()
	print("After running the episodes, T = Target, W = Wall")
	TD.render()
	
  

		
	








