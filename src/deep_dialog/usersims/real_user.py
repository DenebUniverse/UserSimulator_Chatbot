import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse, json, random, copy
from deep_dialog.usersims.usersim import UserSimulator
from deep_dialog import dialog_config


class RealUser(UserSimulator):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set

    def initialize_episode(self):
        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        input_nl = input("user: ")
        response_action = self.nlu_model.generate_dia_act(input_nl)
        response_action['nl'] = input_nl
        return response_action

    def next(self, system_action):
        """ Generate next User Action based on last System Action """
        self.state['turn'] += 2
        self.episode_over = False

        input_nl = input("user:")
        response_action = self.nlu_model.generate_dia_act(input_nl)
        response_action['turn'] = self.state['turn']
        response_action['nl'] = input_nl

        sys_act = system_action['diaact']
        if sys_act == 'closing' or sys_act == 'thanks':
            self.episode_over = True
        return response_action, self.episode_over
