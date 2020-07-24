import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CNet(nn.Module):

    def __init__(self, state_size, fc1_size, fc2_size, action_size):
        super(A2CNet, self).__init__()

        super(A2CNet, self).__init__()

        # detect if  cuda avaialable
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # shared state->fc1->fc2->
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        # actor layer
        self.actor = nn.Linear(fc2_size, action_size)
        # critic layer
        self.critic = nn.Linear(fc2_size, 1)
        # a uniform distribution generator
        self.std = torch.ones(action_size).to(self.device)
        self.dist = torch.distributions.Normal

        # make this run on device

        self.to(self.device)

    def forward(self, s):
        # main body forward pass
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        return s

    def get_action(self, s):
        # get current action with noise
        s = self.forward(s)
        act_mu = self.actor(s)

        dist_var = self.dist(act_mu, self.std)
        action = dist_var.sample()

        # return both the tanh and output layter
        return torch.tanh(action), action

    def get_log_prob(self, s, a):
        s = self.forward(s)
        act_mu = self.actor(s)
        dist_var = self.dist(act_mu, self.std)
        log_prob = dist_var.log_prob(a)
        log_prob = torch.sum(log_prob, dim=1, keepdim=False)
        ent = dist_var.entropy()
        return log_prob, ent

    def get_state_value(self, s):
        s = self.forward(s)
        value = self.critic(s).squeeze(1)
        return value

    def randomize_weights(self):

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.actor.weight.data)
        nn.init.xavier_normal_(self.critic.weight.data)

        nn.init.normal_(self.fc1.bias.data)
        nn.init.normal_(self.fc2.bias.data)
        nn.init.normal_(self.actor.bias.data)
        nn.init.normal_(self.critic.bias.data)        
