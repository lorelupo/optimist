'''
    This script helps creating and managing experiments.
    Possible commands:
    - launch: launch an experiment loading its specification from a CSV file
    - view: list the experiments which are still running
    - stop: stop all the runners of the experiment
'''

import pandas as pd
import argparse, os, sys, re
from multiprocessing import Pool
from screenutils import Screen, list_screens

class Screener(object):

    def command_sender(self, zipped_pair):
        screen, command = zipped_pair
        screen.send_commands(command)

    def run(self, commands, name='s'):
        n_screens = len(commands)
        screens = [Screen(name+'_%d' % (i+1), True) for i in range(n_screens)]

        p = Pool(n_screens)
        p.map(self.command_sender, zip(screens, commands))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--command', help='Command to execute.', type=str, default='launch', choices=['launch', 'view', 'stop'])
# Experiment selection
parser.add_argument('--name', help='Name of the experiment', type=str, default=None)
parser.add_argument('--dir', help='Directory from which to load the experiment (to launch).', type=str, default=None)
# Env
parser.add_argument('--condaenv', help='Conda environment to activate.', type=str, default=None)
parser.add_argument('--pythonv', help='Python version to use', type=str, default='python3')
parser.add_argument('--pythonpath', help='Pythonpath to use for script.', type=str, default=None)
parser.add_argument('--cuda_devices', help='CUDA visible devices.', type=str, default='')
# Sacred
parser.add_argument('--sacred', action='store_true', default=False, help='Enable sacred.')
parser.add_argument('--sacred_dir', help='Dir used by sacred to log.', type=str, default=None)
parser.add_argument('--sacred_slack', help='Config file for slack.', type=str, default=None)
parser.add_argument('--dirty', action='store_true', default=False, help='Enable sacred dirty running.')
args = parser.parse_args()

if args.command == 'launch':
    assert args.name is not None, "Provide an experiment name."
    assert args.dir is not None, "Provide a directory to load the experiment."
    # Load experiment
    experiment_path = args.dir + '/' + args.name + '.csv'
    experiment = pd.read_csv(experiment_path)
    # Start build base command
    cmd_base = ''
    # Set env variables
    cmd_base += 'export CUDA_VISIBLE_DEVICES=' + args.cuda_devices + ' && '
    if args.sacred_dir and args.sacred:
        cmd_base += 'export SACRED_RUNS_DIRECTORY=' + args.sacred_dir + ' && '
    if args.sacred_slack and args.sacred:
        cmd_base += 'export SACRED_SLACK_CONFIG=' + args.sacred_slack + ' && '
    if args.pythonpath:
        cmd_base += "export PYTHONPATH='PYTHONPATH:" + args.pythonpath + "' && "
    if args.condaenv:
        cmd_base += 'source activate ' + args.condaenv + ' && '
    # Parse the CSV
    param_cols = list(experiment)
    param_cols.remove('script')
    # Build the commands
    cmd_base += args.pythonv + ' '
    cmds = []
    for index, row in experiment.iterrows():
        # Get the script, check if we need to use sacred (just append _sacred to script name)
        script = row['script']
        if args.sacred:
            script += '_sacred'
        script = 'baselines/' + script + '.py '
        _c = cmd_base + script
        # Check if dirty and if to use with
        if args.sacred and args.dirty:
            _c += '-e '
        if args.sacred and len(param_cols) > 0:
            _c += 'with '
        # Params
        for p in param_cols:
            if args.sacred:
                _c += str(p).strip() + '=' + str(row[p]).strip() + ' '
            else:
                _c += '--' + str(p).strip() + '=' + str(row[p]).strip() + ' '
        # Add the exit command to terminate the experiment
        _c += '&& exit'
        cmds.append(_c)
    scr = Screener()
    scr.run(cmds, name=args.name)

elif args.command == 'view':
    raise Exception('TBD')

elif args.command == 'stop':
    assert args.name is not None, "Provide an experiment name."
    rule = re.compile(args.name + '_*')
    # Get all screens
    for s in list_screens():
        if rule.match(s.name):
            print("Stopping", s.name)
            s.kill()

else:
    raise Exception('Unrecognized command.')
