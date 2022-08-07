#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np,torch

def model(x,params):
    l1,b1,l2,b2,l3,b3 = params
    y = torch.nn.functional.linear(x,l1,b1)
#     print("YY",y)
    y=torch.relu(y)
    y = torch.nn.functional.linear(y,l2,b2)
    y=torch.relu(y)
    y = torch.nn.functional.linear(y,l3,b3)
    y=torch.log_softmax(y,dim=0)
#     print("Y",y)
    return y

def unpack(params,layers=[(128,8),(64,128),(4,64)]):
    unpacked = []
    
    e=0
    for i,l in enumerate(layers):
#         print("il",i,l)
        s,e = e,e+np.prod(l)
#         print("s,e",s,e)
        weights = params[s:e].view(l)
        s,e= e,e+l[0]
        bias = params[s:e]
        unpacked.extend([weights,bias])
#     print("unpacked",unpacked)
    return unpacked


# In[2]:


def spaw_pop(n=50,size=9668):
    pop =[]
    for i in range(n):
        vec = torch.randn(size)
        fit=0
        p={'params':vec,'fitness':fit}
        pop.append(p)
    return pop


# In[3]:


def recombine(x1,x2):
    x1=x1['params']
    x2=x2['params']
    l=x1.shape[0]
    split_pt = np.random.randint(l)
    child1 = torch.zeros(l)
    child2 = torch.zeros(l)
    child1[0:split_pt] = x1[0:split_pt]
    child1[split_pt:] = x2[split_pt:]
    child2[0:split_pt]=x2[0:split_pt]
    child1[split_pt:] = x1[split_pt:]
    
    c1 = {'params':child1,'fitness':0.0}
    c2 = {'params':child2,'fitness':0.0}
    
    return c1,c2


# In[4]:


def mutate(x,rate=0.01):
    x_ = x['params']
    num_to_change = int(rate*x_.shape[0])
    idx = np.random.randint(low=0,high=x_.shape[0],size=(num_to_change,))
    x_[idx] = torch.randn(num_to_change) / 10.0
    x['params'] = x_
    return x


# In[ ]:


import gym

env = gym.make("LunarLander-v2")

def test_model(agent):
    done = False
    state = torch.from_numpy(env.reset()).float()
    score = 0
    
    while not done:
        params = unpack(agent['params'])
        probs = model(state,params)
        action = torch.distributions.Categorical(probs=probs).sample()
        state_,reward,done,info = env.step(action.item())
        state = torch.from_numpy(state_).float()
        score += 1
    return score
def evaluate_pop(pop):
    tot_fit = 0
    lp = len(pop)
    for agent in pop:
        score = test_model(agent)
        agent['fitness'] = score
        tot_fit += score
        
    avg_fit = tot_fit /lp
    return pop,avg_fit

def next_gen(pop,mut_rate=0.001,tournament_size=0.2):
    new_pop = []
    lp = len(pop)
    while len(new_pop) < len(pop):
        rids = np.random.randint(low=0,high=lp, \
                                size =(int(tournament_size*lp)))
        batch = np.array([[i,x['fitness']] for \
                        (i,x) in enumerate(pop) if i in rids])
        scores = batch[batch[:, 1].argsort()]
        i0,i1 = int(scores[-1][0]),int(scores[-2][0])
        parent0,parent1 = pop[i0],pop[i1]
        offspring_ =  recombine(parent0,parent1)
        child1 = mutate(offspring_[0],rate = mut_rate)
        child2 = mutate(offspring_[1],rate = mut_rate)
        offspring = [child1,child2]
        new_pop.extend(offspring)
    return new_pop
num_gen = 25
pop_size = 800
mut_rate =0.01
pop_fit = []
pop = spaw_pop(n=pop_size,size=9668)
sums = 0
for i in range(num_gen):
    sums += 1
    print(sums)
    pop,avg_fit = evaluate_pop(pop)
    pop_fit.append(avg_fit)
    pop = next_gen(pop,mut_rate=mut_rate,tournament_size=0.2)  
print(sums)


# In[ ]:


from matplotlib import pyplot as plt
def running_mean(x,N):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y
plt.figure(figsize=(12,7))
plt.xlabel("Generations",fontsize=22)
plt.ylabel("Score",fontsize=22)
plt.plot(running_mean(np.array(pop_fit),3))



# In[ ]:





# In[ ]:




