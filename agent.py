import os
import numpy as np
import torch
from envhelper import UnityEnvHelper
# import NN models
from models import A2CNet


class TheTrainer:

    def __init__(self, config ):

        self.config = config
        self.env = None

    def __del__(self):
        
        if self.env != None:
            del self.env
            self.env = None
        
    def start(self, viewer=False , seed = 12345 ):

        # try to set the seed
        torch.manual_seed( seed ) 
        np.random.seed( seed)
        # create a unity helper
        self.env = UnityEnvHelper(self.config.AGENT_FILE , seed=seed , no_graphics=not viewer )
        # create the model
        self.model = A2CNet(self.env.state_size, self.config.FC1_SIZE, self.config.FC2_SIZE, self.env.action_size)
        # reset scores 
        self.scores = []


    def end(self):
        # close environment
        self.env.uenv.close() 
        pass

    def save_model(self, filename='model.pt'):
        # save model weights
        try:
            if self.model != None:
                torch.save(self.model.state_dict(), filename)
                print(f'Model saved as {filename}')
            else:
                raise
        except:
            print('Failed to save model')

    def load_model(self, filename='model.pt'):
        # load weights into model
        if self.model != None:
            if os.path.exists(filename):
                self.model.load_state_dict(torch.load(filename))
            else:
                print(f'\n model not found {filename}')

            # continuous train loop

    def train(self, max_episodes, save_model_filename ):

        # 
        model = self.model 

        # randomize weights before training

        model.randomize_weights() 
        optimizer = torch.optim.Adam(model.parameters(), self.config.LR )

        # create space for saving state / action / rewards 
        batch_s = self.config.N_STEPS * [None]
        batch_a = self.config.N_STEPS * [None]
        batch_r = self.config.N_STEPS * [None]


        batch_vt = self.config.N_STEPS * [None]

        self.scores = []

        for ith in range(1, max_episodes + 1):

            # start an episode
            # reset each agents score
            agt_score = np.zeros(self.env.num_agents)
            # reset n-step batch pointer b
            b = 0
            states = self.env.reset(True)
            while True:
                # evaluate current state(s) 
                # the actions
                model.eval()
                with torch.no_grad():
                    s = torch.from_numpy(states).float().to(model.device)
                    actions_tanh, actions = model.get_action(s)
                model.train()
                # perform the action 
                obs = self.env.step(states, actions_tanh.cpu().data.numpy())
                # update agent scores
                rewards = obs['rewards']
                agt_score += rewards
                # save batch state / reward / action
                batch_s[b] = states
                batch_r[b] = rewards
                batch_a[b] = actions.cpu().data.numpy()
                b += 1

                # state is now the next_state
                states = obs['next_states']
                # our we done?
                done_flag = np.any(obs['dones'])

                # if we have enought samples for N_STEPS ( or about to finish )
                if done_flag or b >= self.config.N_STEPS:

                    # get the current state value 
                    model.eval()
                    s = torch.from_numpy(states).float().to(model.device)
                    rtns = model.get_state_value(s).detach().cpu().data.numpy()
                    model.train()

                    # zero returns where we are done
                    rtns = rtns * (1 - np.array(obs['dones']).astype(float))
                    # calculate discounted returns by going backward through the
                    for i in range(b - 1, -1, -1):
                        rtns = batch_r[i] + self.config.GAMMA * rtns
                        batch_vt[i] = rtns

                    
                    # create a batch of N_AGENTS * N_STEPS , set of states
                    s_batch = torch.from_numpy(np.vstack(batch_s)).float().to(model.device)
                    # and actions
                    a_batch = torch.from_numpy(np.vstack(batch_a)).float().to(model.device)
                    # get the state value for this batch of states 
                    v_batch = model.get_state_value(s_batch)
                    # get the log_probabilities ( and entropy ) of performing each action ob state.
                    log_probs, ent = model.get_log_prob(s_batch, a_batch)

                    # convert the n_step returns into a N_AGENTS * N_STEPS vector

                    vt_batch = torch.from_numpy(np.hstack(batch_vt)).float().to(model.device)
                    # calculate td error 
                    td = vt_batch - v_batch
                    # calculate value loss
                    v_loss = td.pow(2).mean()
                    # calculte policy loss
                    a_loss = -((log_probs * td.detach()).mean())
                    # calculate total loss
                    total_loss = v_loss - ent.mean() * self.config.ENTROPY_WEIGHT + a_loss
                    
                    # update the model to minimize this total_loss.
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    # reset the batch
                    b = 0

                if done_flag:
                    break

            # average score over each agent total reward
            avg_agent_score = agt_score.mean()
            # save score for this episode in list 
            self.scores.append(avg_agent_score)
            # have we got 100 episode scores
            if len(self.scores) >= 100:
                last_100_avg = np.mean(self.scores[-100:])  # mean over last 100 
            else:
                last_100_avg = 0
            
            print(f"\r{ith:5d} ,  {avg_agent_score:8.4f} {last_100_avg:8.4f}", end='' if (ith % 10) else '\n')
            if (last_100_avg >= 30.0):
                # goal found - save 
                print(f"\nEnvironment Solved After {(ith - 100):5d} ")
                self.save_model(save_model_filename)
                break

        return self.scores


    def play(self, max_play_episodes=1 , train_mode = True , max_steps = 100000  ):

        model = self.model
        self.scores = []

        for ith in range(1, max_play_episodes + 1):

            agt_score = np.zeros(self.env.num_agents)
            states = self.env.reset( train_mode )

            steps = 0 
            while steps < max_steps:

                steps+=1 

                model.eval()
                with torch.no_grad():
                    actions_tanh, actions = model.get_action(torch.from_numpy(states).float().to(model.device))
                model.train()

                obs = self.env.step(states, actions_tanh.cpu().data.numpy())

                agt_score += obs['rewards']
                
                if np.any(obs['dones']):
                	break 

                states = obs['next_states']

            self.scores.append( agt_score.mean()  ) 
      

