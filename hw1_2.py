import numpy as np
import matplotlib.pyplot as plt


class bandit(object):
    def __init__(self, arms, T):
        self.arms = arms
        self.T = T
        self.histroy = np.zeros(shape=(self.T, self.arms))
        self.opt_reward = np.zeros(self.T)
        
        self.mu = np.random.normal(0, 1, self.arms)
        print(f'mu: {self.mu}')
        # self.cur_time = 0

    def get_rewards(self, t):   
        R = np.random.normal(self.mu, 1, self.arms)
        self.histroy[t] = R
        opt_arm = np.argmax(self.mu)
        self.opt_reward[t] = R[opt_arm]
        return R

    

class algorithm(object):
    def __init__(self, arms, T):
        self.arms = arms
        self.T = T
        self.Q = np.zeros(shape=(self.T+1, self.arms))
        self.arm_cnt = np.zeros(self.arms)
        self.history = np.zeros(self.T)
        self.arm_chosen = np.zeros(self.T)

    def update_Q(self, arm, reward, t):
        # print(self.Q.shape, arm, reward, t)
        self.Q[t+1][arm] = ((self.Q[t][arm]*self.arm_cnt[arm])+reward)/(self.arm_cnt[arm]+1)
        self.arm_cnt[arm] += 1
        self.history[t] = reward



class Greedy(algorithm):
    def __init__(self, arms, T, N):
        super().__init__(arms, T)
        self.N = N

    def pull_arm(self, t):
        return np.argmax(self.Q[t]) if t<N else


class E_Greedy(algorithm):
    def __init__(self, arms, T, e):
        super().__init__(arms, T)
        self.e = e

    def pull_arm(self, t):
        e_test = np.random.uniform(0,1)
        if e_test > 1-self.e:
            return np.argmax(self.Q[t])
        else:
            return np.random.randint(0, self.arms)

class UCB(algorithm):
    def __init__(self, arms, T, c):
        super().__init__(arms, T)
        self.c = c

    def pull_arm(self, t):
        ucb = self.Q[t] + self.c*np.sqrt( np.log(t)/(self.arm_cnt+0.0001) )
        return np.argmax(ucb)


class Gradient(algorithm):
    def __init__(self, arms, T, a):
        super().__init__(arms, T)
        self.a = a
        self.r_sum = 0

    def update_Q(self, arm, reward, t):
        self.r_sum += reward 
        Q_buf = self.Q[t][arm]
        self.Q[t+1] = self.Q[t] - self.a*(reward - self.r_sum/(t+1))*self.soft_max(t)
        # print(self.a*(self.r_sum/(t+1))*(1-self.soft_max()[arm]))
        self.Q[t+1][arm] = Q_buf + self.a*(reward - self.r_sum/(t+1))*(1-self.soft_max(t)[arm])
        self.history[t] = reward

    def soft_max(self, t):
        return np.exp(self.Q[t])/np.sum(np.exp(self.Q[t]))

    def pull_arm(self, t):
        return np.random.choice(self.arms, 1, p=self.soft_max(t))[0]


if __name__ == '__main__':
    arms = 10
    T = 10000
    np.random.seed(0)
    b = bandit(arms, T)

    g = Greedy(arms, T)
    eg = E_Greedy(arms, T, 0.1)
    ucb = UCB(arms, T, 0.5)
    d = Gradient(arms, T, 0.1)

    for t in range(T):
        r = b.get_rewards(t)
        
        a = g.pull_arm(t)
        g.arm_chosen[t] = a
        g.update_Q( a, r[a], t )

        a = eg.pull_arm(t)
        eg.arm_chosen[t] = a
        eg.update_Q( a, r[a], t )
        
        a = ucb.pull_arm(t)
        ucb.arm_chosen[t] = a
        ucb.update_Q( a, r[a], t )
        
        a = d.pull_arm(t)
        d.arm_chosen[t] = a
        d.update_Q( a, r[a], t )
    
    fig, ax1 = plt.subplots()
    # ax1.set_yscale('log')
    ax1.plot(np.arange(T), (b.opt_reward-g.history).cumsum(), label='greedy')
    ax1.plot(np.arange(T), (b.opt_reward-eg.history).cumsum(), label='e-greedy')
    ax1.plot(np.arange(T), (b.opt_reward-ucb.history).cumsum(), label='UCB')
    ax1.plot(np.arange(T), (b.opt_reward-d.history).cumsum(), label='gradient')
    ax1.legend()
    print(b.mu.argmax())

    # fig, ax2 = plt.subplots()
    # ax2.plot(np.arange(T), g.arm_chosen, label='greedy')
    # ax2.plot(np.arange(T), eg.arm_chosen, label='e-greedy')
    # ax2.plot(np.arange(T), ucb.arm_chosen, label='UCB')
    # ax2.plot(np.arange(T), d.arm_chosen, label='gradient')
    # ax2.legend()

    # plt.legend()
    plt.show()