import argparse, json, copy, os
import pickle
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentDQN
from deep_dialog.usersims import RealUser
from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--diaact_nl_pairs', dest='diaact_nl_pairs', type=str,
                        default='./deep_dialog/data/dia_act_nl_pairs.v6.json',
                        help='path to the pre-defined dia_act&NL pairs')

    parser.add_argument('--max_turn', dest='max_turn', default=20, type=int,
                        help='maximum length of each dialog (default=20, 0=no maximum length)')
    parser.add_argument('--episodes', dest='episodes', default=1, type=int,
                        help='Total number of episodes to run (default=1)')
    parser.add_argument('--slot_err_prob', dest='slot_err_prob', default=0.05, type=float,
                        help='the slot err probability')
    parser.add_argument('--slot_err_mode', dest='slot_err_mode', default=0, type=int,
                        help='slot_err_mode: 0 for slot_val only; 1 for three errs')
    parser.add_argument('--intent_err_prob', dest='intent_err_prob', default=0.05, type=float,
                        help='the intent err probability')

    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0,
                        help='Epsilon to determine stochasticity of epsilon-greedy agent policies')

    # load NLG & NLU model
    parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str,
                        default='./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p',
                        help='path to model file')
    parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str,
                        default='./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p',
                        help='path to the NLU model file')

    parser.add_argument('--act_level', dest='act_level', type=int, default=0,
                        help='0 for dia_act level; 1 for NL level')
    parser.add_argument('--run_mode', dest='run_mode', type=int, default=0,
                        help='run_mode: 0 for default NL; 1 for dia_act; 2 for both')
    parser.add_argument('--auto_suggest', dest='auto_suggest', type=int, default=0,
                        help='0 for no auto_suggest; 1 for auto_suggest')
    parser.add_argument('--cmd_input_mode', dest='cmd_input_mode', type=int, default=0,
                        help='run_mode: 0 for NL; 1 for dia_act')

    # RL agent parameters
    parser.add_argument('--experience_replay_pool_size', dest='experience_replay_pool_size', type=int, default=1000,
                        help='the size for experience replay')
    parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=60,
                        help='the hidden size for DQN')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='gamma for DQN')
    parser.add_argument('--predict_mode', dest='predict_mode', type=bool, default=False, help='predict model for DQN')
    parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size', type=int, default=50,
                        help='the size of validation set')
    parser.add_argument('--warm_start', dest='warm_start', type=int, default=1,
                        help='0: no warm start; 1: warm start for training')
    parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=100,
                        help='the number of epochs for warm start')

    parser.add_argument('--trained_model_path', dest='trained_model_path', type=str,
                        default='./deep_dialog/checkpoints/agt_9_287_340_0.89200.p',
                        help='the path for trained model')  # trained_model_path None将训练
    parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str,
                        default='./deep_dialog/checkpoints/', help='write model to disk')
    parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=10,
                        help='number of epochs for saving model')

    parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float, default=0.3,
                        help='the threshold for success rate')

    parser.add_argument('--split_fold', dest='split_fold', default=5, type=int,
                        help='the number of folders to split the user goal')
    parser.add_argument('--learning_phase', dest='learning_phase', default='all', type=str,
                        help='train/test/all; default is all')

    args = parser.parse_args()
    params = vars(args)

################################################################################
#   导入数据
################################################################################

# 电影数据库
movie_kb_path = './deep_dialog/data/movie_kb.1k.p'
with open(movie_kb_path, "rb") as f:
    movie_kb = pickle.load(f, encoding="latin1")

# 意图集合
act_set = text_to_dict('./deep_dialog/data/dia_acts.txt')
# 槽位集合
slot_set = text_to_dict('./deep_dialog/data/slot_set.txt')
# slot所有可能取值,simulator用
dict_path = './deep_dialog/data/dicts.v3.p'
movie_dictionary = pickle.load(open(dict_path, 'rb'))

max_turn = 20  # 最长对话轮数
num_episodes = 1  # 对话个数
################################################################################
#   加载模型
################################################################################
# Agents参数
agent_params = {}
agent_params['max_turn'] = max_turn  # 最长对话轮数
agent_params['epsilon'] = params['epsilon']  # 随机生成回复的概率
agent_params['agent_run_mode'] = params['run_mode']
agent_params['agent_act_level'] = params['act_level']
agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['cmd_input_mode'] = params['cmd_input_mode']

agent = AgentDQN(movie_kb, act_set, slot_set, agent_params)

# 真实用户
user_sim = RealUser(movie_dictionary, act_set, slot_set)

# 加载NLG
nlg_model_path = params['nlg_model_path']
diaact_nl_pairs = params['diaact_nl_pairs']
nlg_model = nlg()
nlg_model.load_nlg_model(nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)

agent.set_nlg_model(nlg_model)
user_sim.set_nlg_model(nlg_model)

# 加载NLU
nlu_model_path = params['nlu_model_path']
nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

agent.set_nlu_model(nlu_model)
user_sim.set_nlu_model(nlu_model)

# 加载对话管理器
dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, movie_kb)
################################################################################
#   开始对话
################################################################################

from deep_dialog.dialog_system import StateTracker


def test_run():
    print('Testing Started...')
    # 开始记录对话状态
    state_tracker = StateTracker(act_set, slot_set, movie_kb)
    state_tracker.initialize_episode()
    # 用户提问开始对话
    user_action = user_sim.initialize_episode()
    agent.initialize_episode()
    episode_over = False
    state_tracker.update(user_action=user_action)
    while not episode_over:
        ########################################################################
        #   DST将状态传递给agent，agent生成动作
        ########################################################################
        state = state_tracker.get_state_for_agent()  # 字典:当前用户行为+已有信息+上轮agent的行为+数据库中匹配的结果数量
        agent_action = agent.state_to_action(state)  # 对状态编码成向量形式的表征，输入MLP预测多分类，做出动作
        ########################################################################
        #   DST根据动作更新agent状态记录
        ########################################################################
        state_tracker.update(agent_action=agent_action)
        agent.add_nl_to_action(agent_action)  # add NL to Agent Dia_Act
        ########################################################################
        #   用户反馈动作的结果（回应、奖励、成功、结束会话）
        ########################################################################
        sys_action = state_tracker.dialog_history_dictionaries()[-1]
        print(agent_action['act_slot_response']['nl'])
        user_action, episode_over = user_sim.next(sys_action)  # 输入
        reward = 0  # 奖励
        ########################################################################
        #   DST更新用户状态记录
        ########################################################################
        if episode_over != True:
            state_tracker.update(user_action=user_action)
        ########################################################################
        #  DST将新一轮状态传递给agent，agent积累本轮动作经验
        ########################################################################
        agent.register_experience_replay_tuple(state, agent_action, reward,
                                               state_tracker.get_state_for_agent(), episode_over)


test_run()
