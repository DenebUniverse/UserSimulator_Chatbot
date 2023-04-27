from deep_dialog import dialog_config


class Agent:
    """ Agent原型 """

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        """
        参数
        movie_dict      --  数据
        act_set         --  所有可能的动作/意图
        slot_set        --  所有可能的槽位
        params          --  其他参数
        """
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']

    def initialize_episode(self):
        """ 每次对话前初始化 """
        self.current_action = {}
        self.current_action['diaact'] = None
        self.current_action['inform_slots'] = {}
        self.current_action['request_slots'] = {}
        self.current_action['turn'] = 0

    def state_to_action(self, state, available_actions):
        """ 根据对话状态生成行为 """
        act_slot_response = None
        act_slot_value_response = None
        return {"act_slot_response": act_slot_response, "act_slot_value_response": act_slot_value_response}

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """  Register feedback from the environment, to be stored as future training data

        Arguments:
        s_t                 --  The state in which the last action was taken
        a_t                 --  The previous agent action
        reward              --  The reward received immediately following the action
        s_tplus1            --  The state transition following the latest action
        episode_over        --  A boolean value representing whether the this is the final action.

        Returns:
        None
        """
        pass

    def set_nlg_model(self, nlg_model):
        """ 调用nlg模型 """
        self.nlg_model = nlg_model

    def set_nlu_model(self, nlu_model):
        """ 调用nlu模型"""
        self.nlu_model = nlu_model

    def add_nl_to_action(self, agent_action):
        """ 行为转换为自然语言 """
        if agent_action['act_slot_response']:
            agent_action['act_slot_response']['nl'] = ""
            user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_response'], 'agt')
            agent_action['act_slot_response']['nl'] = user_nlg_sentence
        elif agent_action['act_slot_value_response']:
            agent_action['act_slot_value_response']['nl'] = ""
            user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_value_response'], 'agt')
            agent_action['act_slot_response']['nl'] = user_nlg_sentence
