import numpy as np
from physics_sim import PhysicsSim
#from agent import ReplayBuffer #for reward function

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
            
            def get_propeler_thrust(self, rotor_speeds): 
                '''calculates net thrust (thrust - drag) based on velocity
                of propeller and incoming power'''
                thrusts = []
        
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        #You can use only [z_pos, z_speed], so it would be much easier for the agent to learn and train.
        #[self.sim.pose[2], self.sim.v[2]]
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6 #making state size larger
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4 #4 rotors?
        
        #for reward function
        self.z_bonus = 5.0 #tried 5 but I think a huge reward is better like 10

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) #txyz so this is takeoff
        #z_target_pos = self.target_pos[2]
##################################################################################################        
    """def computeDistance(self, current_pos, target_pos): #helper function for get_reward
        distance = []
        for i in range(len(current_pos)):
            axis_dist=current_pos[i]-target_pos[i]
            distance.append(axis_dist)
        return distance """   
        
############################################################################reward function
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0
        #original
        #reward += (1.-.3*(abs(self.sim.pose[2] - self.target_pos[2])))#if 11 then 0.7, then 1.9
        
        ###########from review##########################################################################
        z_diff =  self.sim.pose[2]- self.target_pos[2] #negative means current lower than target
        z_factor = self.z_bonus if z_diff >= 0 else 5.0 #z_bonus is 5.0

        # The closer the better the reward
        if z_diff<=0: #if current is below target
            distance = abs(z_diff) 
            reward += (1 / distance) * z_factor
        else: #if higher than target
            distance = z_diff
            reward += distance * z_factor
        
        #Punish any descent from initial z of 2
        up_or_down = self.sim.pose[2] - 2 #if descent is neg, it descended
        reward+= up_or_down 
        #to reward vertical velocity#####################################################################
        vx = self.sim.v[0]
        vy = self.sim.v[1]
        vz = self.sim.v[2] 
        if vz <= 0:
            reward-=1
        
        if vz>=0: #the faster this is, the higher the reward
            reward += vz
       
        #toReturn = np.tanh(reward) #normalize return to [-1,1]
        
        
        return reward 
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
             
            reward += self.get_reward() 
        ##########################################################################################################
            # penalize crash
            #if done and self.sim.time < self.sim.runtime: 
             #   reward = -1
        #############################################################################################################  
             
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state