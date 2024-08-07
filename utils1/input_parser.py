import argparse
from argparse import Namespace
from datetime import datetime
from pathlib import Path


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


class Parser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse(self) -> Namespace:
        parser_ = argparse.ArgumentParser()
        parser_.add_argument('-lr', '--lr_schedule', default=1e-4, type=float, help='learning rate')
        parser_.add_argument('-e', '--episode_count', default=30, type=int, help='number of episode to train')
        parser_.add_argument('-ee', '--epochs', default=100, type=int, help='number of episode to train')
        parser_.add_argument('-sc', '--steps_count', default=4096, type=int, help='number of steps per episode')
        parser_.add_argument('-m', '--servers_cnt', default=25, type=int, help='number of servers')
        parser_.add_argument('-k', '--demanded_func', default=2, type=int, help='number of functions')
        parser_.add_argument('-f', '--possible_func', default=5,  type=int, help='number of functions')
        parser_.add_argument('-n', '--clients_cnt', default=500,  type=int, help='number of clients')
        parser_.add_argument('-w', '--w1', default=5, type=int, help='lower limit for weights range')
        parser_.add_argument('-ww', '--w2', default=12, type=int, help='upper limit for weights range')
        parser_.add_argument('-r', '--radius', default=15, type=int, help='percentage of diameter in the graph')
        args = parser_.parse_args()
        return args
##Approcate ratio for client-server-demand##
## servercount = 25, function = 5, client = 500, demanded function = 2, weight = 5-12, radius = 15# --> total demand == 1000. 1000 < total served < 1250 ->
##