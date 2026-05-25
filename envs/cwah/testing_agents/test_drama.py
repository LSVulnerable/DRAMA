import sys
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
print("Working directory:", curr_dir)
sys.path.append(f'{curr_dir}/..')
sys.path.append(f'{curr_dir}/../..')
import ipdb
import pickle
import json
import random
import numpy as np
import threading
import time

from pathlib import Path
from envs.unity_environment import UnityEnvironment
from agents import LLM_agent
from arguments import get_args
from algos.arena_mp2 import ArenaMP
import logging
from LLM import *
from agents import *


if __name__ == "__main__":
	
	args = get_args()
	if args.new_join and args.join_num > 0:
		args.agent_num += args.join_num
	args.mode = f"{args.mode}_env{args.test_task}_{args.agent_num}agents_{args.lm_id_list}_org{args.organization_code}" + ("_comm" if args.comm else "") + ("_random_start_comm" if args.random_start_comm else "")
	
	# Load LLM configs
	llm_configs_dict = json.load(open('./llm_configs.json','r'))
	if args.lm_id_list is None and args.llm_config_list is None:
		raise NotImplementedError
	elif len(args.lm_id_list) == 1:
		args.lm_id_list = [args.lm_id_list[0]] * args.agent_num
	else:
		assert len(args.lm_id_list) == args.agent_num, "lm_id_list must be the same length as agent_num"
	
	if args.llm_config_list is None:
		args.llm_config_list = []
		for lm_id in args.lm_id_list:
			if lm_id == 'human':
				args.llm_config_list.append(None)
			else:
				llm_config = llm_configs_dict[lm_id]
				llm_config.update({'seed': args.seed})
				args.llm_config_list.append(llm_config)
	print("args.lm_id_list:", args.lm_id_list)
	
	env_task_set = pickle.load(open(args.dataset_path, 'rb'))

	logging.basicConfig(format='%(asctime)s - %(name)s:\n %(message)s', level=logging.INFO)
	File_handler = logging.FileHandler(f'{args.log_path}{args.mode}.log', encoding="utf-8")
	logger = logging.getLogger(__name__)
	logger.addHandler(File_handler)

	if args.organization_code is not None:
		df = pd.read_csv('../testing_agents/organization_instructions.csv')
		try:
			args.organization_instructions = df['instruction'][int(args.organization_code)]
		except:
			print("Invalid organization code")
			raise NotImplementedError

	args.record_dir = f'../test_results/{args.mode}' 
	logger.info("mode: {}".format(args.mode))
	Path(args.record_dir).mkdir(parents=True, exist_ok=True)

	if "image" in args.obs_type or args.gen_video:
		os.system("Xvfb :94 & export DISPLAY=:94")
		import time
		time.sleep(3) # ensure Xvfb is open
		os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
		executable_args = {
						'file_name': args.executable_file,
						'x_display': '94',
						'no_graphics': False,
						'timeout_wait': 5000,
		}
	else:
		executable_args = {
						'file_name': args.executable_file,
						'no_graphics': True,
						'timeout_wait': 500,
		}

	id_run = 0
	random.seed(args.seed) #id_run
	episode_ids = list(range(len(env_task_set)))
	episode_ids = sorted(episode_ids)
	num_tries = 1 #numbers to repeat all the tasks
	num_runs = args.num_runs #number of runs for each task
	S = [[] for _ in range(len(episode_ids))]
	L = [[] for _ in range(len(episode_ids))]
	
	def env_fn(env_id):
		return UnityEnvironment(num_agents=args.agent_num,
							   max_episode_length=args.max_episode_length,
							   port_id=env_id,
							   env_task_set=env_task_set,
							   agent_goals=['LLM' for i in range(args.agent_num)],
							   observation_types=[args.obs_type for i in range(args.agent_num)],
							   use_editor=args.use_editor,
							   executable_args=executable_args,
							   base_port=args.base_port if args.base_port is not None else np.random.randint(11000, 13000),
							   seed=args.seed,
							   recording_options={'recording': True if args.gen_video else False,
									'output_folder': args.record_dir,
									'file_name_prefix': args.mode,
									'cameras': 'PERSON_FROM_BACK',
									'modality': 'normal'}
							   )
	
	agents = [lambda x, y: None for i in range(args.agent_num)]

	for iter in range(num_tries):
		steps_list, failed_tasks = [], []
		comm_tokens, comm_times = [], []
		monitor_problems, monitor_suggestions = [], []
		if not os.path.isfile(args.record_dir + '/results.pik'):
			test_results = {}
		else:
			test_results = pickle.load(open(args.record_dir + '/results.pik', 'rb'))

		current_tried = iter
		episode_ids = episode_ids[args.start_task:args.end_task + 1] if args.start_task is not None and args.end_task is not None else episode_ids
 
		for episode in episode_ids:
			for iter in range(num_runs):
				for _ in range(5):
					try:
						arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents, args.record_dir, args.debug, run_predefined_actions=False, comm=args.comm, args=args, type='drama', disconnected=args.disconnected, new_join=args.new_join)
						break
					except Exception as e:
						np.random.seed(int(time.time()))
						logger.error(f"Error initializing arena: {e}")
						time.sleep(5)
						continue

				logger.info('episode: {}'.format(episode + 1))
				is_finished = 0
				steps = 250
				arena.reset(episode, reset_seed=args.seed + int(time.time()))

				success, steps, saved_info, total_targets_num = arena.run()
				collision_rate = (arena.env.collision_num / steps ) if steps > 0 else 0
				throughput = (total_targets_num / steps ) if steps > 0 else 0

				logger.info('------------------')
				logger.info('success' if success else 'failure')
				logger.info('steps: {}'.format(steps))
				logger.info('collision_rate: {:.4f}'.format((arena.env.collision_num / steps ) if steps > 0 else 0))
				logger.info('throughput: {:.4f}'.format((total_targets_num / steps ) if steps > 0 else 0))
				arena.comm_cost_info['converse']['comm_tokens_per_step'] = round(arena.comm_cost_info['converse']['overall_tokens'] / steps, 2)
				arena.comm_cost_info['converse']['comm_times_per_step'] = round(arena.comm_cost_info['converse']['overall_times'] / steps, 2)
				logger.info('comm_cost_info: {}'.format(arena.comm_cost_info))
				logger.info('------------------')

				if not success:
					failed_tasks.append(episode)
					logger.info('failed_tasks: {}'.format(failed_tasks))
					continue
				else:
					steps_list.append(steps)
					comm_tokens.append(arena.comm_cost_info['converse']['overall_tokens'])
					comm_times.append(arena.comm_cost_info['converse']['overall_times'])
				is_finished = 1 if success else 0
				log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(saved_info['task_id'],
																					saved_info['task_name'],
																					current_tried)

				if len(saved_info['obs']) > 0:
					pickle.dump(saved_info, open(log_file_name, 'wb'))
				else:
					with open(log_file_name, 'w+') as f:
						f.write(json.dumps(saved_info, indent=4))

				S[episode].append(is_finished)
				L[episode].append(steps)

				test_results[episode] = {'S': S[episode],
											'L': L[episode]}
				
				arena.close()
				arena = None
				logger.info('episode {} finished'.format(episode + 1))
				
				logger.info("mode: {}".format(args.mode))
				logger.info('average steps (finishing the tasks): {:.2f}'.format(np.array(steps_list).mean() if len(steps_list) > 0 else None))
				logger.info('collision_rate (finishing the tasks): {:.4f}'.format(collision_rate))
				logger.info('throughput (finishing the tasks): {:.4f}'.format(throughput))
				logger.info('input_usage (finishing the tasks): {:.2f}'.format(saved_info.get('input_usage', 0)))
				logger.info('output_usage (finishing the tasks): {:.2f}'.format(saved_info.get('output_usage', 0)))

				logger.info('failed_tasks: {}'.format(failed_tasks))
				time.sleep(10)

		pickle.dump(test_results, open(args.record_dir + '/results.pik', 'wb'))
