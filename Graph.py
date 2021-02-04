from glob import glob
import matplotlib.pyplot as plt
import numpy as np


def score(moving_avg, name=" "):
    x = []
    y = []
    # plt.figure(dpi=300)
    if '-' in name:
        title = name.split('-')[0]
    elif '_' in name:
        title = name.split('_')[0]
    elif ' ' in name:
        title = name.split(' ')[0]
    else:
        title = name
    plt.title(title)
    [(x.append(i), y.append(j)) for i, j in enumerate(moving_avg)]
    plt.scatter(x, y, marker='|', c=[-i for i in y])
    plt.ylabel(" Score")
    plt.xlabel("Number of Games Played")
    plt.show()
    plt.savefig(name + " Score.png", transparent=True)
    plt.close()


def average(moving_avg, name=" ", n=10):
    x = []
    y = []
    sum_ = 0
    for i, j in enumerate(moving_avg):
        sum_ += j
        if i % n == 0:
            x.append(i)
            y.append(sum_ / n)
            sum_ = 0
    # plt.figure(dpi=300)
    if '-' in name:
        title = name.split('-')[0]
    elif '_' in name:
        title = name.split('_')[0]
    elif ' ' in name:
        title = name.split(' ')[0]
    else:
        title = name
    plt.title(title)
    plt.plot(x, y, '-')
    plt.ylabel("Average over {:d} Episodes".format(n))
    plt.xlabel("Number of Games Played")
    plt.show()
    plt.savefig(name + "_AvgOver" + str(n) + ".png", transparent=True)
    plt.close()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_single(data, ylabel, title):
    plt.figure()
    plt.plot(data)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.close()


def comparison(Sarsa, DQN, DDQN, dueling, DoubleDueling, ylabel, title):
    plt.figure()
    plt.plot(range(len(Sarsa)), Sarsa, range(len(DQN)), DQN, range(len(DDQN)), DDQN, range(len(dueling)), dueling,
             range(len(DoubleDueling)), DoubleDueling)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(['Deep Sarsa', 'DQN', 'DDQN', 'Dueling', 'Double Dueling'])
    plt.show()
    plt.close()


def main():
    file1 = "Logs/log_SpaceInvaders_Sarsa.txt"
    file2 = "Logs/log_SpaceInvaders_DQN.txt"
    file3 = "Logs/log_SpaceInvaders_DDQN.txt"
    file4 = "Logs/log_SpaceInvaders_Dueling.txt"
    file5 = "Logs/log_SpaceInvaders_DDQN+Dueling.txt"

    # file = glob("log.txt")
    # file = file[int(input(
    #     str([x for x in zip(range(1, file.__len__() + 1), file)])[2:-2].replace("), (", "\n").replace(',', ':')
    #     + '\n\nEnter File Number: ')) - 1]
    avgSarsa = []
    for line in open(file1, 'r').readlines(-1):
        try:
            avgSarsa.append(float((line.rsplit('\t', 4)[1])[14:]))
        except:
            pass
    print('\nNo. of Episodes = ', avgSarsa.__len__(), '\nAverage Score = {:0.2f}'.format(sum(avgSarsa) / avgSarsa.__len__()))

    avgDQN = []
    for line in open(file2, 'r').readlines(-1):
        try:
            avgDQN.append(float((line.rsplit('\t', 4)[1])[14:]))
        except:
            pass
    print('\nNo. of Episodes = ', avgDQN.__len__(),
          '\nAverage Score = {:0.2f}'.format(sum(avgDQN) / avgDQN.__len__()))

    avgDDQN = []
    for line in open(file3, 'r').readlines(-1):
        try:
            avgDDQN.append(float((line.rsplit('\t', 4)[1])[14:]))
        except:
            pass
    print('\nNo. of Episodes = ', avgDDQN.__len__(),
          '\nAverage Score = {:0.2f}'.format(sum(avgDDQN) / avgDDQN.__len__()))

    avgDueling = []
    for line in open(file4, 'r').readlines(-1):
        try:
            avgDueling.append(float((line.rsplit('\t', 4)[1])[14:]))
        except:
            pass
    print('\nNo. of Episodes = ', avgDueling.__len__(),
          '\nAverage Score = {:0.2f}'.format(sum(avgDueling) / avgDueling.__len__()))

    avgDoubleDueling = []
    for line in open(file5, 'r').readlines(-1):
        try:
            avgDoubleDueling.append(float((line.rsplit('\t', 4)[1])[14:]))
        except:
            pass
    print('\nNo. of Episodes = ', avgDoubleDueling.__len__(),
          '\nAverage Score = {:0.2f}'.format(sum(avgDoubleDueling) / avgDoubleDueling.__len__()))

    while True:
        c = int(input('\n\n1. Plot Score vs Episode\n2. Plot Avgerage over n Episodes\n3. All\n0. Exit\n\n'))

        if c == 1:
            score(avgSarsa, file1[:-4])
            score(avgDQN, file2[:-4])
            score(avgDDQN, file3[:-4])
            score(avgDueling, file4[:-4])
            score(avgDoubleDueling, file5[:-4])
        elif c == 2:
            forEpisode = int(input('\nEnter number of Episodes to average: '))
            average(avgSarsa, file1[:-4], forEpisode)
            average(avgDQN, file2[:-4], forEpisode)
            average(avgDDQN, file3[:-4], forEpisode)
            average(avgDueling, file4[:-4], forEpisode)
            average(avgDoubleDueling, file5[:-4], forEpisode)
        elif c == 0:
            return None
        elif c == 3:
            reward_Sarsa = moving_average(avgSarsa[:4200], 100)
            plot_single(reward_Sarsa, 'avg reward', 'average reward Deep Sarsa')

            reward_DQN = moving_average(avgDQN[:4200], 1)
            plot_single(reward_DQN, 'avg reward', 'average reward DQN')

            reward_DDQN = moving_average(avgDDQN[:4200], 10)
            plot_single(reward_DDQN, 'avg reward', 'average reward DDQN')

            reward_Dueling = moving_average(avgDueling[:4200], 1)
            plot_single(reward_Dueling, 'avg reward', 'average reward Dueling')

            reward_DoubleDueling = moving_average(avgDoubleDueling[:4200], 1)
            plot_single(reward_DoubleDueling, 'avg reward', 'average reward Double Dueling')

            reward_Sarsa = moving_average(avgSarsa[:4200], 50)
            reward_DQN = moving_average(avgDQN[:4200], 50)
            reward_DDQN = moving_average(avgDDQN[:4200], 50)
            reward_Dueling = moving_average(avgDueling[:4200], 50)
            reward_DoubleDueling = moving_average(avgDoubleDueling[:4200], 50)
            comparison(reward_Sarsa, reward_DQN, reward_DDQN, reward_Dueling, reward_DoubleDueling,  'Average reward',
                       'Reward comparison between Deep Sarsa, DQN, DDQN, dueling DQN, double dueling DQN')


if __name__ == '__main__':
    main()
