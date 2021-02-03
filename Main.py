import gym
from PIL import Image
import numpy as np
from Agent import Agent
import sys
import gc
from time import time, sleep
import os


def main(game, dqn, ddqn, dueling, both):
    # print(dqn, ddqn, dueling, both)
    env_name = game + "-v0"
    print(env_name)

    algo = ""
    if dqn:
        algo = "_DQN"
    elif ddqn:
        algo = "_DDQN"
    elif dueling:
        algo = "_Dueling"
    elif both:
        ddqn = True
        dueling = True
        algo = "_DDQN+Dueling"
        
    if not os.path.exists("Experiences"):
        os.mkdir("Experiences")
    if not os.path.exists("Logs"):
        os.mkdir("Logs")
    if not os.path.exists("Weights"):
        os.mkdir("Weights")

    log = open("Logs/log_" + game + algo + ".txt", 'a')
    episodes = int(input("Episodes? : "))
    timer = time()

    env = gym.make(env_name)

    agent = Agent((84, 84, 4), env.action_space.n, env_name, load_weights=True, ddqn=ddqn, dueling=dueling)
    count = 0
    _ = 0
    av_score = 0
    min_experiences = 0 if agent.load_state(True) else 1500
    for episode in range(episodes):
        obsv = process_state(env.reset())
        current_state = np.array([obsv, obsv, obsv, obsv])
        # print("obs")
        score = 0
        done = False
        steps = _
        while not done:
            _ += 1
            env.render() if episodes - episode - 1 <= 3 else None

            action = agent.action(np.asarray([current_state]))
            # print("Action Taken: ", action)

            obsv, reward, done, info = env.step(action)
            obsv = process_state(obsv)
            next_state = get_next_state(current_state, obsv)

            clipped_reward = np.clip(reward, -1, 1)
            # print("Clipped Reward: ", clipped_reward)

            agent.experience_gain(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

            if agent.experience_available() and _ > min_experiences:
                agent.greedy()
                if _ % 4 == 0:  # trenuj každé 4 framy
                    count += 1
                    agent.train()
                    print("Steps : ", _, "\tCount = ", count)
                    if count == 800:
                        count = 0
                        agent.save()
                        agent.update_target_network()

            current_state = next_state
            score += reward
        steps = _ - steps

        agent.save()
        env.close()
        timer = time() - timer
        av_score = (av_score + score) / 2 if episode != 0 else score
        print(episode + 1, "\tTotalReward = ", score, "\tSteps: ", steps, "\tMoving Avg: {:.2f}".format(av_score),
              "\tTime: %d" % (timer / 60), "\b:{:.0f}".format((timer % 60)))
        log.write(str(episode + 1) + "\tTotalReward = " + str(score) + "\tSteps: " + str(
            steps) + "\tMoving Avg: {:.2f}".format(av_score) + "\tTime: %d" % int(timer / 60) + ":{:.0f} \n".format(
            (timer % 60)))
        timer = time()

        # print(episode)
        if episode % 10 == 0 and episode != 0 and episode != episodes - 1:
            print("Continuous save")
            agent.save_state(True)

    print("\n\nTotal Steps = ", _)
    log.close()

    del timer
    del current_state
    del next_state
    del count
    del _
    del steps
    del log
    del env
    gc.collect()

    agent.save_state(False)


def test(game, dqn, ddqn, dueling, both):
    algo = ""
    if dqn:
        algo = "DQN"
    elif ddqn:
        algo = "DDQN"
    elif dueling:
        algo = "Dueling"
    elif both:
        ddqn = True
        dueling = True
        algo = "DDQN+Dueling"

    print("Testing " + algo)

    env_name = game + "-v0"
    env = gym.make(env_name)

    agent = Agent((84, 84, 4), env.action_space.n, env_name, epsilon=0, load_weights=True, test=True,
                  ddqn=ddqn, dueling=dueling)

    run = 'y'
    while run == 'y' or run == 'Y':
        try:
            obsv = process_state(env.reset())
            current_state = np.array([obsv, obsv, obsv, obsv])
            done = False
            score = _ = reward = 0
            while not done:
                _ += 1
                env.render()
                # sleep(0.01)
                action = agent.action(np.asarray([current_state]))
                # print(action)
                # Image.fromarray(process_state(env.render("rgb_array"))).show() if _ % 500 == 0 else None
                # print(action) if reward != 0 else None

                obsv, reward, done, info = env.step(action)
                obsv = process_state(obsv)
                next_state = get_next_state(current_state, obsv)

                current_state = next_state
                score += reward

            print("Total Reward: ", score, "\nSteps: ", _)
            env.close()
        except KeyboardInterrupt:
            env.step(env.action_space.sample())
        run = input("\nRUN TEST AGAIN? (Y/N) : ")
    # print("Exiting Environment.")
    env.close()


def process_state(observation):
    img = np.asanyarray(Image.fromarray(observation, 'RGB').convert('L').resize((84, 84)))
    # img = np.delete(img, np.s_[-13:], 0)
    # img = np.delete(img, np.s_[:13], 0)
    # img = np.delete(img, np.s_[-10:], 0)
    # img = np.delete(img, np.s_[:16], 0)
    # print(img.shape)
    # Image._show(Image.fromarray(img))

    return img

    # Image._show(Image.fromarray(img))
    # Image._show((Image.open(observation).convert('L').resize((84, 110))))
    # Image._showxv(Image.fromarray(observation, 'RGB').convert('L').resize((84, 110)), "BEFORE CROPPING")


def get_next_state(current, observation):
    return np.append(current[1:], [observation], axis=0)


# def processed_screen():
#     name = "SpaceInvaders-v0"
#     env = gym.make(name)
#     env.reset()
#     for i in range(400):
#         env.step(env.action_space.sample())
#     screen = env.render("rgb_array")
#     Image.fromarray(screen).save(name + ".png")
#     Image.fromarray(process_state(screen)).show()


if __name__ == '__main__':
    # processed_screen()
    try:
        game_name = sys.argv[1]
    except IndexError:
        game_name = "SpaceInvaders"
    if input("1. Train Agent\n2. Run Test\n\n: ") == '1':
        alg = input("1. DQN\n2. DDQN\n3. Dueling\n4. Both\n\n: ")
        if alg == '1':
            main(game_name, True, False, False, False)
        elif alg == "2":
            main(game_name, False, True, False, False)
        elif alg == "3":
            main(game_name, False, False, True, False)
        elif alg == "4":
            main(game_name, False, False, False, True)

    else:
        alg = input("1. DQN\n2. DDQN\n3. Dueling\n4. Both\n\n: ")
        if alg == '1':
            test(game_name, True, False, False, False)
        elif alg == "2":
            test(game_name, False, True, False, False)
        elif alg == "3":
            test(game_name, False, False, True, False)
        elif alg == "4":
            test(game_name, False, False, False, True)
        # test(game_name)
        # process_state("SC0.png")
        # processed_screen()
