from unityagents import UnityEnvironment

'''
    A very simple / basic class to abstract the UnityEnivronment even further
'''


class UnityEnvHelper:

    # constructor - give file_name of agent environment

    def __init__(self, file_name, no_graphics=True, seed=8888):

        self.seed = seed
        self.uenv = UnityEnvironment(file_name=file_name, seed=self.seed, no_graphics=no_graphics )

        # pick the first agent as the brain

        self.brain_name = self.uenv.brain_names[0]
        self.brain = self.uenv.brains[self.brain_name]

        # get the action space size 

        self.action_size = self.brain.vector_action_space_size

        # reset the environment , in training mode 

        self.reset(True)

        # get the state space size
        self.state_size = len(self.ue_info.vector_observations[0])

        self.num_agents = len(self.ue_info.agents)

    def __del__(self):

        # make sure we close the environment 
        pass

    def reset(self, train_mode=True):

        # tell the unity agent to restart an episode 
        # training mode simple seems to run the simulation at full speed 
        self.ue_info = self.uenv.reset(train_mode=train_mode)[self.brain_name]
        return self.ue_info.vector_observations

    # we pass in current state for convenience 
    def step(self, state, action):

        # perform action on environment  and get observation
        self.ue_info = self.uenv.step(action)[self.brain_name]
        # return state , action , next state , reward and done flag
        # slightly 
        return {'states': state,
                'actions': action,
                'rewards': self.reward(),
                'next_states': self.state(),
                'dones': self.done()}

    def state(self):
        # just last observation state
        return self.ue_info.vector_observations

    def reward(self):
        # return reward from last observation
        return self.ue_info.rewards

    def done(self):
        # return done flag  
        return self.ue_info.local_done
