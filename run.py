from tools import make_stage, set_seed
from mario import MarioAgent

if __name__ == '__main__':
    # stage
    stage_f, stage_s, stage_t, stage_four = make_stage()
    # ex para
    episodes = 10 #40000
    sync_interval = 10 #1000
    swa_start = 4 #2000
    c = 100
    # preparing for ex
    set_seed()
    agent = MarioAgent(c)
    env = stage_f
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_loss = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            loss = agent.update(state, action, reward, next_state, done, swa_start, episode)
            state = next_state
            total_loss += loss
        if episode % sync_interval == 0:
            agent.update_target_qnet()
        
        epsi = agent.update_epsilon()
        total_reward1 = agent.eval_stage(1)
        total_reward2 = agent.eval_stage(2)

    if episode >= swa_start:
        if episode % c == 0:
            print('【 サンプル抽出: エピソード {} 】'.format(episode))
        total_reward_swa1 = agent.eval_swa(1)
        total_reward_swa2 = agent.eval_swa(2)

        if episode % 1 == 0:
            print('【 エピソード:{} 】 [ DQN 累積報酬:{} ] [ DQN with SWA 累積報酬:{} ]  [ (stage1-2) DQN 累積報酬:{} ] [ (stage1-2) DQN with SWA 累積報酬:{} ] [ ロス:{} ]'\
                .format(episode, total_reward1, total_reward_swa1, total_reward2, total_reward_swa2, total_loss))
        #wandb.log({"Episode": episode, "Reward_1-1": total_reward1, "Reward_1-1_swa": total_reward_swa1,  "Reward_1-2": total_reward2, "Reward_1-2_swa": total_reward_swa2, "loss":total_loss, "Epsilon":epsi})
    else:
        if episode % 1 == 0: 
            print('【 エピソード:{} 】 [ DQN 累積報酬:{} ] [ DQN with SWA 累積報酬:{} ]  [ (stage1-2) DQN 累積報酬:{} ] [ (stage1-2) DQN with SWA 累積報酬:{} ] [ ロス:{} ]'\
                .format(episode, total_reward1,  total_reward1, total_reward2, total_reward2, total_loss))
        #wandb.log({"Episode": episode, "Reward_1-1": total_reward1, "Reward_1-1_swa": total_reward1,  "Reward_1-2": total_reward2, "Reward_1-2_swa": total_reward2, "loss":total_loss, "Epsilon":epsi})

    if episode+1 == swa_start:
        print('【 SWA Start 】')        
# normal
total_reward_1 = agent.eval_stage(1)
total_reward_2 = agent.eval_stage(2)
total_reward_3 = agent.eval_stage(3)
total_reward_4 = agent.eval_stage(4)

# SWA
total_reward_swa_1 = agent.eval_swa(1)
total_reward_swa_2 = agent.eval_swa(2)
total_reward_swa_3 = agent.eval_swa(3)
total_reward_swa_4 = agent.eval_swa(4)