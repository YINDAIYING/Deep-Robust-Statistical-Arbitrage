import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import json
import time
import os
import random
import seaborn as sns
import sys

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

df = pd.read_csv("^STOXX50E.csv")

stoxx = df.Close
stoxx_dates = df.Date

stoxx = stoxx.dropna()
stoxx_dates = stoxx_dates.dropna()

stoxx_dates = stoxx_dates[2087:len(stoxx)-78].values
stoxx = stoxx[2087:len(stoxx)-78].values

test_begin_day = 229*21
test_end_day = (286+1)*21

asset_list = [stoxx]
z = 1 #number of assets

plt.figure()
for i in range(z):
  plt.plot(asset_list[i])

def get_slice_array1(i):
  temp = np.tile(np.arange(i+1), z)
  return temp+np.repeat(np.arange(z),i+1)*1

def get_slice_array2():
  temp = np.tile(np.arange(1,2), z)
  return temp+np.repeat(np.arange(z),1)*3

def get_slice_array3():
  temp = np.tile(np.arange(1,3), z)
  return temp+np.repeat(np.arange(z),2)*3

def beta(x):
    return torch.pow(F.relu(x),2)

def get_random_partition(stock_mins, stock_maxs, num_sets, discrete = True):
    partition = np.zeros((num_sets, 2*z))
    stocks_len = (stock_maxs-stock_mins)/grids
    for k in range(num_sets):
      if discrete:
        lower = np.random.randint(0, grids, size=z)
        gap = np.random.randint(1, grids-lower+1, size=z)
        partition[k,:] = np.repeat(stock_mins,2)
        partition[k,::2] += lower*stocks_len
        partition[k,1::2] += (lower+gap)*stocks_len
      else:
        lower = np.random.uniform(low = stock_mins, high = stock_maxs)
        upper = stock_maxs
        partition[k,::2] = lower
        partition[k,1::2] = upper
    return partition

def get_indicator_matrix(partition, S_n):
    B = partition.shape[0]
    indicator_matrix = np.zeros((len(S_n), 2**B))
    for i in range(len(S_n)):
      temp = np.ones(B, dtype=bool)
      for j in range(z):
        temp = temp&(partition[:,2*j]<S_n[i,j])
      temp = temp.flatten()
      temp = temp.astype(int)
      pos = int(np.array2string(temp, formatter={'int':lambda x: bin(x)})[1:-1].replace(' ','')[2::3],2)
      indicator_matrix[i][pos]=1
    return torch.FloatTensor(indicator_matrix)

def get_frac_vector(P_tau_tilde, indicator_matrix):
    #get input data for neural network
    net_input = P_tau_tilde[:,get_slice_array2()]
    #perform forward propagation to obtain deltas for each asset
    delta_list, c = net(net_input)
    h_value_trans = 0
    h_value_pnl = 0
    #calculate the loss function summation
    for i in range(z):
      delta = delta_list[i]
      h_value_trans += trans_cost * torch.abs(torch.cat((delta[:,:1], delta[:,1:]-delta[:,:-1], delta[:,-1:]), 1)).sum(axis = 1)
      h_value_pnl += ((P_tau_tilde[:,3*i+1:3*i+3]-P_tau_tilde[:,3*i:3*i+2])*delta).sum(axis = 1)
    h_value = -c + h_value_trans - h_value_pnl
    temp = torch.matmul(indicator_matrix.T, h_value)
    indicator_sum = torch.sum(indicator_matrix.T, dim=1)
    indicator_sum = torch.where(indicator_sum.abs() > 0.5, indicator_sum, torch.ones((), device = indicator_sum.device, dtype = indicator_sum.dtype))
    return temp/indicator_sum

def stat_arb_success(stock_test, net):

    f = 0
    gain_list = []
    cost_list = []
    success = []
    
    for i in range(int(len(stock_test[0])/22)):
        
        query_data = np.zeros((1, 1*z))

        for j in range(z):
          query_data[0,1*j:1*(j+1)] = (100*stock_test[j][22*i+11:22*i+22:11]/stock_test[j][22*i]).reshape(1,-1)

        upper_bound = net(torch.FloatTensor(query_data).cuda())[0]
        upper_bound_list = [delta.detach().cpu().numpy()[0] for delta in upper_bound]

        gain = 0
        cost = 0
        for k in range(z):

          delta = upper_bound_list[k]
          gain += np.dot(delta, stock_test[k][22*i+11:22*i+22:10]-stock_test[k][22*i:22*i+22:11])
          cost += trans_cost * np.abs(np.concatenate([np.array([delta[0]]), delta[1:]-delta[:-1], np.array([delta[-1]])])).sum()

        success.append(gain-cost)
        gain_list.append(gain)
        cost_list.append(cost)

        f+=success[-1]
    
    print("Toal Gain:", np.array(gain_list).mean(), "Total Cost:", np.array(cost_list).mean())
    success = np.array(success)
    res = dict()
    res['Gain'] = f
    res['Best'] = round(max(success),2)
    res['Worst'] = round(min(success),2)
    res['Average'] = round(success.mean(),2)
    res['Std'] = round(success.std(),2)
    res['loss_perc'] = round(100*float(np.sum(success<0)/len(success)),2)
    res['gains_perc'] = round(100*float(np.sum(success>0)/len(success)),2)
    res['sharp_ratio'] = round(np.sqrt(252.0/21)*success.mean()/success.std(),3)
    res['sortino_ratio'] = round(np.sqrt(252.0/21)*success.mean()/success[success<0].std(),3)
    return res

def stat_arb_success_buy_and_hold(stock_test):

    f = 0
    gain_list = []
    cost_list = []
    success = []
    
    for i in range(int(len(stock_test[0])/22)):

        gain = 0
        cost = 0

        for k in range(z):

          delta = 1*np.ones(2)
          gain += np.dot(delta, stock_test[k][22*i+11:22*i+22:10]-stock_test[k][22*i:22*i+22:11])
          cost += trans_cost * np.abs(np.concatenate([np.array([delta[0]]), delta[1:]-delta[:-1], np.array([delta[-1]])])).sum()
        
        success.append(gain-cost)
        gain_list.append(gain)
        cost_list.append(cost)

        f+=success[-1]
    
    print("Toal Gain:", np.array(gain_list).mean(), "Total Cost:", np.array(cost_list).mean())
    success = np.array(success)
    res = dict()
    res['Gain'] = f
    res['Best'] = round(max(success),2)
    res['Worst'] = round(min(success),2)
    res['Average'] = round(success.mean(),2)
    res['Std'] = round(success.std(),2)
    res['loss_perc'] = round(100*float(np.sum(success<0)/len(success)),2)
    res['gains_perc'] = round(100*float(np.sum(success>0)/len(success)),2)
    res['sharp_ratio'] = round(np.sqrt(252.0/21)*success.mean()/success.std(),3)
    res['sortino_ratio'] = round(np.sqrt(252.0/21)*success.mean()/success[success<0].std(),3)
    return res

class Net_independent(nn.Module):

    def __init__(self, M, z):
        super(Net_independent, self).__init__()
        self.first_layers = nn.ModuleList([nn.Linear(z*i,32*z) for i in range(1,1+1)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(z*i) for i in range(1,1+1)])
        self.second_layers = nn.ModuleList([nn.Linear(32*z,64*z) for i in range(1,1+1)])
        self.third_layers = nn.ModuleList([nn.Linear(64*z,128*z) for i in range(1,1+1)])
        self.fourth_layers = nn.ModuleList([nn.Linear(128*z,z) for i in range(1,1+1)])
        self.c = nn.Parameter(torch.FloatTensor([0]))
        self.delta0_s = nn.Parameter(torch.FloatTensor([0]*z))
        self.M = M

    def forward(self, x):
        output1 = [F.relu(self.first_layers[i](self.bn_layers[i](x[:,get_slice_array1(i)]))) for i in range(1)]
        output2 = [F.relu(self.second_layers[i](output1[i])) for i in range(1)]
        output3 = [F.relu(self.third_layers[i](output2[i])) for i in range(1)]
        deltas = [M*torch.tanh(self.fourth_layers[i](output3[i])) for i in range(1)]
        delta_list = [torch.cat([delta[:,i:i+1] for delta in deltas], axis=1) for i in range(z)]
        return [torch.cat((self.delta0_s[i]*torch.ones((x.shape[0],1)).cuda(),delta_list[i]), axis=1) for i in range(z)], self.c

stock_mins = np.array([np.min(asset[:test_begin_day]) for asset in asset_list])
stock_maxs = np.array([np.max(asset[:test_begin_day]) for asset in asset_list])

#radius refers to the lower and upper bounds centered at 100
#e.g. radius = 50 means the bounds are [100-50,100+50]
radius = 50
rescaled_lower_bounds = np.array([100-radius]*z)
rescaled_upper_bounds = np.array([100+radius]*z)

stock_train = [asset[:test_begin_day] for asset in asset_list]
stock_test = [asset[test_begin_day:test_end_day] for asset in asset_list]

N = len(stock_train[0])
n=3-1

P_hat = np.zeros((z*(n+1),N-21))

for j in range(z):
  for i in range(n+1):
    if i==0:
      P_hat[i+j*(n+1),:] = stock_train[j][:N-21]
    elif i==1:
      P_hat[i+j*(n+1),:] = stock_train[j][11:N-21+11]
    elif i==2:
      P_hat[i+j*(n+1),:] = stock_train[j][21:N-21+21]

P_hat = torch.FloatTensor(P_hat.T)

results = {'Average':[], 'Std':[], 'Best':[], 'Worst':[], 'loss_perc':[], 'gains_perc':[], 'sharp_ratio':[], 'sortino_ratio':[]}

num_epochs = 100
N_measures = 5
grids = 4
K = 1
trials = 50
epsilon = z
M = 1
trans_cost = 0
pnl = []

#print result of buy and hold strategy
print(stat_arb_success_buy_and_hold(stock_test))

for trial in range(trials):
  np.random.seed(trial)
  torch.manual_seed(trial)
  print("Trial:", trial)
  net = Net_independent(M,z).cuda()
  net.train()
  optimizer = optim.Adam(net.parameters(), lr=1e-3)
  loss_rec = []
  c_rec = []
  for epoch in range(num_epochs):

      if epoch%10==0:
        print("Test at Epoch", epoch)
        net.eval()
        record = stat_arb_success(stock_test, net)
        net.train()
        print(record)

      optimizer.zero_grad()
      total_loss = 0
      partition = get_random_partition(rescaled_lower_bounds, rescaled_upper_bounds, 12, False)
      
      for m in range(N_measures):
          
          batch_index = np.arange(P_hat.shape[0])

          P_tau_tilde = P_hat.clone()

          for j in range(z):
            P_tau_tilde[:,3*j:3*(j+1)] /= (P_tau_tilde[:,3*j]/100).reshape(-1,1)

          tau_m = torch.FloatTensor(np.random.normal(0,1,(P_hat.shape[0], P_hat.shape[1]-z)))
          U_epsilon = torch.FloatTensor(epsilon * np.random.rand(1))
          tau_tilde = U_epsilon * tau_m/torch.norm(tau_m, p=2, dim=1, keepdim=True)

          P_tau_tilde[:,get_slice_array3()] += tau_tilde

          S_n = P_tau_tilde[:,np.arange(1,z+1)*3-1].numpy()
          S_n = S_n[batch_index]

          indicator_matrix = get_indicator_matrix(partition, S_n).cuda()
          P_tau_tilde = P_tau_tilde.cuda()
          P_tau_tilde = P_tau_tilde[batch_index]

          frac_vector = get_frac_vector(P_tau_tilde, indicator_matrix)
          total_loss += beta(torch.matmul(indicator_matrix, frac_vector)).sum()/len(S_n)

      total_loss = net.c + K * total_loss
      loss_rec.append(total_loss.item())
      c_rec.append(net.c.clone().detach().cpu().numpy()[0])
      total_loss.backward()
      optimizer.step()
      with torch.no_grad():
        net.c.clamp_(-M, M)
        net.delta0_s.clamp_(-M, M)

  print("Done Training")
  net.eval()
  record = stat_arb_success(stock_test, net)
  print("Result After Training:")
  print(record)

  pnl.append(record)

  plt.figure()
  plt.plot(loss_rec)
  plt.title('Loss Function')
  plt.savefig("loss.eps", format = 'eps')

  plt.figure()
  plt.plot(c_rec)
  plt.title('c variable')
  plt.savefig("c.png")

plt.hist([p['Average'] for p in pnl])

print("Overall Profit:", np.round(np.mean([result['Gain'] for result in pnl]),2))
print("Average Profit:", np.round(np.mean([result['Average'] for result in pnl]),2))
print("Profit Std:", np.round(np.mean([result['Std'] for result in pnl]),2))
print("gains_perc:", np.round(np.mean([result['gains_perc'] for result in pnl]),2))
print("Best:", np.round(np.mean([result['Best'] for result in pnl]),2))
print("Worst:", np.round(np.mean([result['Worst'] for result in pnl]),2))
print("sharp_ratio:", np.round(np.mean([result['sharp_ratio'] for result in pnl]),4))
print("sortino_ratio:", np.round(np.mean([result['sortino_ratio'] for result in pnl]),4))
