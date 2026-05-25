import os
import pdb
import pickle
import random
import torch
import copy
import numpy as np
from tqdm import tqdm
import time
import ipdb
import json
import atexit
import sys
import pandas as pd
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{curr_dir}')
from comms import *
from heartbeat import *
from collections import defaultdict, Counter
from agents import LLM_agent
import logging
logger = logging.getLogger("__main__")

import threading
import time
from typing import Dict, List
import re
from scipy.optimize import linear_sum_assignment


# @ray.remote
class ArenaMP(object):
	def __init__(self, max_number_steps, arena_id, environment_fn, agent_fn, record_dir='out', debug=False, run_predefined_actions=False, comm=False, args=None, disconnected=False, new_join=False, type=None):
		# run_predefined_actions is a parameter that you can use predefined_actions.json to strictly set the agents' actions instead of using algorithm to calculate the action.

		self.agents = {}
		self.agent_names = ["Agent_{}".format(i+1) for i in range(len(agent_fn))]
		self.comm = comm
		self.disconnected = disconnected
		self.new_join = new_join
		self.env_fn = environment_fn
		self.agent_fn = agent_fn
		self.arena_id = arena_id
		self.num_agents = args.agent_num
		self.random_start_comm = args.random_start_comm
		self.log_thoughts = args.log_thoughts
		self.vis_comm = not args.no_comm_fig
		self.fig_path = f'{args.log_path}{args.mode}'
		self.organization_instructions = args.organization_instructions
		self.action_history_len = args.action_history_len
		self.dialogue_history_len = args.dialogue_history_len
		self.task_goal = None
		self.record_dir = record_dir
		self.debug = debug
		self.prompt_template_path = args.prompt_template_path

		self.heartbeat_manager = None
		self.subgoals = {}
		self.prev_satisfied = {}
		self.reassign = False
		self.reexecute = False
		self.fail = False
		self.type = type
		if self.type == 'drama':
			self.init_run = False
		if self.type == 'agentverse':
			self.evaluate = False

		self.satisfied_disconnect_done = False
		self.satisfied_newjoin_done = False
		self.turnover = False
		if self.disconnected and self.new_join:
			self.turnover = True
			self.disconnected = False
			self.new_join = False

		self.disconnect_count = args.dis_num
		if self.disconnect_count == 2:
			self.double_disconnect = True
		else:
			self.double_disconnect = False
		self.new_join_count = args.join_num
		if self.new_join_count == 2:
			self.double_join = True
		else:
			self.double_join = False
		self.turnover_same = False

		logger.info("Init Env")
		self.env = environment_fn(arena_id)
		self.converse_agents = []
		self.comm_cost_info = {
					"converse": {"overall_tokens": 0, "overall_times": 0},
					"select": {"tokens_in": 0, "tokens_out": 0}
				}
		for i in range(self.num_agents):
			if self.type == 'mcts':
				from agents.MCTS_agent import MCTS_agent
				self.args_common = dict(recursive=False,
					   max_episode_length=5,
					   num_simulation=100,
					   max_rollout_steps=5,
					   c_init=0.1,
					   c_base=1000000,
					   num_samples=1,
					   num_processes=1,
					   logging=True,
					   logging_graphs=True,
					   opponent_subgoal=args.opponent_subgoal,
					   belief_comm=args.belief_comm,
					   satisfied_comm=args.satisfied_comm
					   )
				agent = {"agent_id": i+1, 'char_index': i}
				agent.update(self.args_common)
				agent = MCTS_agent(**agent)
				self.agents[i] = agent
			else:
				agent = LLM_agent(agent_id=i+1, args=args)
				self.agents[i] = agent

				converse_agent = AssistantAgentCoT(
					self.agent_names[i],
					system_message="",
					llm_config=self.agents[i].LLM.llm_config,
					max_consecutive_auto_reply=1, # we have to manually set maximum conversation turn
					sampling_params=self.agents[i].LLM.sampling_params
				)
				self.converse_agents.append(converse_agent)
				self.comm_cost_info["converse"][self.agent_names[i]] = {"tokens_in": 0, "tokens_out": 0, "times_in": 0, "times_out": 0}
				for j in range(self.num_agents):
					if i != j:
						self.comm_cost_info["converse"][f"{self.agent_names[i]} to {self.agent_names[j]}"] = {"tokens": 0, "times": 0}
		
		self.max_episode_length = self.env.max_episode_length
		self.max_number_steps = max_number_steps
		self.run_predefined_actions = run_predefined_actions
		atexit.register(self.close)

		self.dict_info = {}
		self.dict_dialogue_history = defaultdict(list)
		self.LLM_returns = {}

		self.pause_event = threading.Event()
		self.pause_event.set()  # Initially not paused

		self.task_info = {}
		self.guardian_chain_units = {}

		# ===== Prob-MLP: persistence & training cfg =====
		self.enable_prob_net = False
		self.rooms_order = []
		self.prob_net = None          
		self.prob_opt = None
		self.prob_net_hidden = 64
		self.prob_train = True
		self.prob_lr = 1e-3
		self.prob_ckpt_dir = os.path.join(curr_dir, "./checkpoints")
		self.prob_model_file = os.path.join(self.prob_ckpt_dir, "mlp.pt")
		self.prob_map_file = os.path.join(self.prob_ckpt_dir, "global_prob_maps.json")
		self.global_obj_room_probs = {}
		self._load_prob_assets()

	def _ensure_prob_dirs(self):
		try:
			os.makedirs(self.prob_ckpt_dir, exist_ok=True)
		except Exception:
			pass
	def _build_prob_net(self, nrooms: int):
		self.prob_net = torch.nn.Sequential(
			torch.nn.Linear(2 * nrooms, self.prob_net_hidden),
			torch.nn.ReLU(),
			torch.nn.Linear(self.prob_net_hidden, nrooms),
		)
		self.prob_net.eval()
		self.prob_opt = torch.optim.Adam(self.prob_net.parameters(), lr=self.prob_lr)
	def _load_prob_assets(self):
		self._ensure_prob_dirs()

		try:
			if os.path.exists(self.prob_map_file):
				with open(self.prob_map_file, "r", encoding="utf-8") as f:
					self.global_obj_room_probs = json.load(f)
		except Exception:
			self.global_obj_room_probs = {}

	def _maybe_init_or_load_prob_net(self, rooms: List[str]):

		if not rooms:
			return
		if not self.rooms_order:
			self.rooms_order = list(rooms)
		n = len(self.rooms_order)
		if (self.prob_net is None) or (getattr(self.prob_net[-1], "out_features", None) != n):
			self._build_prob_net(n)

		try:
			if os.path.exists(self.prob_model_file):
				state = torch.load(self.prob_model_file, map_location="cpu")
				if state.get("nrooms") == n:
					self.prob_net.load_state_dict(state["state_dict"])
					self.prob_net.eval()
		except Exception:
			pass
	def _save_prob_model(self):
		self._ensure_prob_dirs()
		try:
			if self.prob_net is not None and self.rooms_order:
				torch.save(
					{
						"state_dict": self.prob_net.state_dict(),
						"nrooms": len(self.rooms_order),
						"rooms_order": self.rooms_order,
					},
					self.prob_model_file,
				)
		except Exception:
			pass
	def _save_global_probs(self):
		self._ensure_prob_dirs()
		try:
			for obj, info in self.task_info.items():
				pmap = info.get("prob")
				if isinstance(pmap, dict) and pmap:
					self.global_obj_room_probs[obj] = {k: float(v) for k, v in pmap.items()}
			with open(self.prob_map_file, "w", encoding="utf-8") as f:
				json.dump(self.global_obj_room_probs, f, ensure_ascii=False, indent=2)
		except Exception:
			pass
	def save_probnet_and_probs(self):
		self._save_prob_model()
		self._save_global_probs()

	def wait_if_paused(self):
		"""Wait until the pause event is set."""
		self.pause_event.wait()
	
	def pause(self):
		"""Pause the execution by clearing the pause event."""
		self.pause_event.clear()

	def resume(self):
		"""Resume the execution by setting the pause event."""
		self.pause_event.set()

	def close(self):
		self.env.close()

	def get_port(self):
		return self.env.port_number

	def reset(self, task_id=None, reset_seed=None):
		if self.heartbeat_manager is not None:
			try:
				self.heartbeat_manager.stop_all_threads()
			except Exception as e:
				logger.info(f"Error stopping heartbeat threads: {e}")
			finally:
				self.heartbeat_manager = None
		self.cnt_duplicate_subgoal = 0
		self.cnt_nouse_subgoal = 0
		self.dict_info = {}
		self.dict_dialogue_history = defaultdict(list)
		self.LLM_returns = {}
		self.converse_agents = []
		self.comm_cost_info = {
					"converse": {"overall_tokens": 0, "overall_times": 0},
					"select": {"tokens_in": 0, "tokens_out": 0}
				}
		for agent in self.agents.values():
			agent.status = 'active'
		if self.type == 'LLM' or self.type == 'drama' or self.type == 'agentverse':
			for i in range(self.num_agents):
				converse_agent = AssistantAgentCoT(
					self.agent_names[i],
					system_message="",
					llm_config=self.agents[i].LLM.llm_config,
					max_consecutive_auto_reply=1, # we have to manually set maximum conversation turn
					sampling_params=self.agents[i].LLM.sampling_params
				)
				self.converse_agents.append(converse_agent)
				self.comm_cost_info["converse"][self.agent_names[i]] = {"tokens_in": 0, "tokens_out": 0, "times_in": 0, "times_out": 0}
				for j in range(self.num_agents):
					if i != j:
						self.comm_cost_info["converse"][f"{self.agent_names[i]} to {self.agent_names[j]}"] = {"tokens": 0, "times": 0}
		if self.run_predefined_actions:
			self.action_notes_steps = 0
			with open("predefined_actions.json","r", encoding='utf-8') as f:
				self.action_notes = json.load(f)
		ob = None
		while ob is None:
			ob = self.env.reset(task_id=task_id, reset_seed=reset_seed)

		for it, agent in self.agents.items():
			if 'LLM_vision' in agent.agent_type:
				agent.reset(ob[it], self.env.all_containers_name, self.env.all_goal_objects_name, self.env.all_room_name, self.env.goal_spec[it])
			elif 'vision' in agent.agent_type:
				agent.reset(ob[it], self.env.full_graph, self.env.task_goal, self.env.all_room_name, self.env.all_containers_name, self.env.all_goal_objects_name, seed=agent.seed)
			elif 'MCTS' in agent.agent_type or 'Random' in agent.agent_type:
				agent.reset(ob[it], self.env.full_graph, self.env.task_goal, seed=agent.seed)
			elif 'LLM' in agent.agent_type:
				agent.reset(ob[it], self.env.all_containers_name, self.env.all_goal_objects_name, self.env.all_room_name, self.env.room_info, self.env.goal_spec[it])
			else:
				agent.reset(self.env.full_graph)

		if (self.new_join or self.turnover)and self.new_join_count > 0:
			for new_agent_id in range(self.num_agents - self.new_join_count, self.num_agents):
				self.agents[new_agent_id].status = 'disconnected'
			self.num_agents -= self.new_join_count

		if self.type == 'drama':
			self._init_obj_info()
			self.assign_tasks_from_affinity()

	def set_weigths(self, epsilon, weights):
		for agent in self.agents.values():
			if 'RL' in agent.agent_type:
				agent.epsilon = epsilon
				agent.actor_critic.load_state_dict(weights)

	def discussion(self, obs):
		df = pd.read_csv(self.prompt_template_path)
		prompt_format = df['prompt'][1]
		selector_prompts = []

		skip_flag = False
		for it, agent in self.agents.items():
			if agent.status == 'disconnected':
				if self.dict_dialogue_history[f"Agent_{it + 1}"] != []:
					self.dict_dialogue_history[f"Agent_{it + 1}"].append([f"[SYSTEM] Agent_{it + 1} is disconnected."])
				continue
			if self.dict_dialogue_history[f"Agent_{it + 1}"] != []:
				history_list = self.dict_dialogue_history[f"Agent_{it + 1}"]
				system_msgs = []
				regular_msgs = []
				for dialogue in history_list:
					if isinstance(dialogue, list) and any('[SYSTEM]' in str(msg) for msg in dialogue):
						system_msgs.append(dialogue)
					else :
						regular_msgs.append(dialogue)

				if len(regular_msgs) > self.dialogue_history_len - len(system_msgs):
					regular_msgs = regular_msgs[-(self.dialogue_history_len - len(system_msgs)):]
				self.dict_dialogue_history[f"Agent_{it + 1}"] = system_msgs + regular_msgs
				dialogue_history = [word for dialogue in self.dict_dialogue_history[f"Agent_{it + 1}"] for word in dialogue]
			else:
				dialogue_history = []

			if self.task_goal is None:
				goal_spec = self.env.get_goal(self.env.task_goal[it], self.env.agent_goals[it])
			else:
				goal_spec = self.env.get_goal(self.task_goal[it], self.env.agent_goals[it])
		
			goal_desc = self.agents[it].LLM.goal_desc
			action_history = self.agents[it].action_history
			action_history = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)

			_ = self.agents[it].obs_processing(obs[it], goal_spec)
			progress = self.agents[it].progress2text()

			teammate_names = []
			for i, agent in enumerate(self.agent_names):
				if i != it and self.agents[i].status != 'disconnected':
					teammate_names.append(agent)
			subgoal_desc = []
			for i, subgoal in self.subgoals.items():
				subgoal_desc.append(f"Agent_{i + 1}'s subgoal is {subgoal}.")
			
			selector_prompt = prompt_format.replace("$AGENT_NAME$", f"Agent_{it + 1}")
			selector_prompt = selector_prompt.replace("$ORGANIZATION_INSTRUCTIONS$", self.organization_instructions)
			selector_prompt = selector_prompt.replace("$TEAMMATE_NAME$", ", ".join(teammate_names))
			selector_prompt = selector_prompt.replace("$GOAL$", str(goal_desc))
			selector_prompt = selector_prompt.replace("$SUBGOAL$", str(subgoal_desc))
			selector_prompt = selector_prompt.replace("$PROGRESS$", str(progress))
			selector_prompt = selector_prompt.replace("$ACTION_HISTORY$", str(action_history))
			selector_prompt = selector_prompt.replace("$DIALOGUE_HISTORY$", str('\n'.join(dialogue_history)))
			selector_prompts.append(selector_prompt)

		if not skip_flag:
			self.dict_dialogue_history, self.comm_cost_info = communicate(self.agent_names, \
				selector_prompts, self.dict_dialogue_history, self.comm_cost_info, list(self.agents.values()), self.converse_agents, self.random_start_comm, self.log_thoughts, visualize=self.vis_comm, fig_folder_name=self.fig_path)

	def get_actions(self, obs, action_space=None, true_graph=False):
		dict_actions = {}
		if self.run_predefined_actions:
			act = self.action_notes[str(self.action_notes_steps)]
			self.action_notes_steps += 1
			split = act.find('|')
			actdict = {0:act[:split], 1:act[split+1:]}
			return actdict, {}

		if self.comm:
			logger.info('Communication at step {}'.format(self.env.steps))
			self.discussion(obs)

			if self.debug:
				logger.info('comm_cost_info: {}'.format(self.comm_cost_info))
			logger.info('------------------')

		logger.info('Actions at step {}'.format(self.env.steps))
		for it, agent in self.agents.items():
			if agent.status == 'disconnected':
				continue
			if self.type != 'mcts':
				logger.info(f'Agent_{it+1}')
			if self.task_goal is None:
				goal_spec = self.env.get_goal(self.env.task_goal[it], self.env.agent_goals[it])
			else:
				goal_spec = self.env.get_goal(self.task_goal[it], self.env.agent_goals[it])
			
			if agent.agent_type in ['MCTS', 'Random', 'MCTS_vision']:
				teammate_subgoal = None
				if agent.recursive:
					teammate_subgoal = self.agents[1 - it].last_subgoal
				dict_actions[it], _ = agent.get_action(obs[it], goal_spec, teammate_subgoal)
				
			elif 'RL' in agent.agent_type:
				if 'MCTS' in agent.agent_type or 'Random' in agent.agent_type:
					if true_graph:
						full_graph = self.env.get_graph()
					else:
						full_graph = None
					dict_actions[it], _ = agent.get_action(obs[it], goal_spec,
																	   action_space_ids=action_space[it], full_graph=full_graph)

				else:
					dict_actions[it], _ = agent.get_action(obs[it], self.task_goal, action_space_ids=action_space[it])

			elif 'LLM' in agent.agent_type:
				obs = self.env.get_observations()
				dict_actions[it], self.dict_info[it] = agent.get_action(obs[it], goal_spec, dialogue_history=self.dict_dialogue_history[f"Agent_{it + 1}"], subgoal = self.subgoals[it] if it in self.subgoals and self.subgoals[it] else None, reexecute = self.reexecute)


		return dict_actions, self.dict_info

	def reset_env(self):
		self.env.close()
		self.env = self.env_fn(self.arena_id)

	def rollout_reset(self, logging=False, record=False, episode_id=None, is_train=True, goals=None, args=None):
		try:
			res = self.rollout(logging, record, episode_id=episode_id, is_train=is_train, goals=goals)
			return res
		except:
			self.env.close()
			self.env = self.env_fn(self.arena_id)

			for agent in self.agents.values():
				if 'RL' in agent.agent_type:
					prev_eps = agent.epsilon
					prev_weights = agent.actor_critic.state_dict()

			self.agents = {}
			for i in range(self.num_agents):
				agent = (LLM_agent(agent_id=i+1, args=args))
				self.agents[i] = agent

			self.set_weigths(prev_eps, prev_weights)
			return self.rollout(logging, record, episode_id=episode_id, is_train=is_train, goals=goals)

	def rollout(self, logging=0, record=False, episode_id=None, is_train=True, goals=None):
		t1 = time.time()
		print("rollout", episode_id, is_train)
		if episode_id is not None:
			self.reset(episode_id)
		else:
			self.reset()

		t2 = time.time()
		t_reset = t2 - t1
		c_r_all = [0] * self.num_agents
		success_r_all = [0] * self.num_agents
		done = False
		actions = []
		nb_steps = 0
		agent_steps = 0
		info_rollout = {}
		entropy_action, entropy_object = [], []
		observation_space, action_space = [], []

		if goals is not None:
			self.task_goal = goals
		else:
			self.task_goal = None

		if logging > 0:
			info_rollout['pred_goal'] = []
			info_rollout['pred_close'] = []
			info_rollout['gt_goal'] = []
			info_rollout['gt_close'] = []
			info_rollout['mask_nodes'] = []
		init_dict = {i: [] for i in range(self.num_agents)}
		if logging > 1:
			info_rollout['step_info'] = []
			info_rollout['action'] = init_dict
			info_rollout['script'] = []
			info_rollout['graph'] = []
			info_rollout['action_space_ids'] = []
			info_rollout['visible_ids'] = []
			info_rollout['action_tried'] = []
			info_rollout['predicate'] = []
			info_rollout['reward'] = []
			info_rollout['goals_finished'] = []
			info_rollout['obs'] = []

		rollout_agent = {}

		for agent_id in range(self.num_agents):
			agent = self.agents[agent_id]
			if 'RL' in agent.agent_type:
				rollout_agent[agent_id] = []

		if logging:
			init_graph = self.env.get_graph()
			pred = self.env.goal_spec[0]
			goal_class = [elem_name.split('_')[1] for elem_name in list(pred.keys())]
			id2node = {node['id']: node for node in init_graph['nodes']}
			info_goals = []
			info_goals.append([node for node in init_graph['nodes'] if node['class_name'] in goal_class])
			ids_target = [node['id'] for node in init_graph['nodes'] if node['class_name'] in goal_class]
			info_goals.append([(id2node[edge['to_id']]['class_name'],
								edge['to_id'],
								edge['relation_type'],
								edge['from_id']) for edge in init_graph['edges'] if edge['from_id'] in ids_target])
			info_rollout['target'] = [pred, info_goals]

		agent_id = [id for id, enum_agent in self.agents.items() if 'RL' in enum_agent.agent_type][0]
		reward_step = 0
		prev_reward_step = 0
		curr_num_steps = 0
		prev_reward = 0
		init_step_agent_info = {}
		local_rollout_actions = []
		if not is_train:
			pbar = tqdm(total=self.max_episode_length)
		while not done and nb_steps < self.max_episode_length and agent_steps < self.max_number_steps:
			(obs, reward, done, env_info), agent_actions, agent_info = self.step(true_graph=is_train)
			step_failed = env_info['failed_exec']
			if step_failed:
				print("FAILING in task")
				print(agent_actions)
				print(local_rollout_actions)
				print('----')
			local_rollout_actions.append(agent_actions[0])
			if not is_train:
				pbar.update(1)
			if logging:
				curr_graph = env_info['graph']
				agentindex = self.agents[agent_id].agent_id
				observed_nodes = agent_info[agent_id]['visible_ids']
				node_id = [node['bounding_box'] for node in obs[agent_id]['nodes'] if node['id'] == agentindex][0]
				edges_char = [(id2node[edge['to_id']]['class_name'],
								edge['to_id'],
								edge['relation_type']) for edge in curr_graph['edges'] if edge['from_id'] == agentindex and edge['to_id'] in observed_nodes]

				if logging > 0:
					if 'pred_goal' in agent_info[agent_id].keys():
						info_rollout['pred_goal'].append(agent_info[agent_id]['pred_goal'])
						info_rollout['pred_close'].append(agent_info[agent_id]['pred_close'])
						info_rollout['gt_goal'].append(agent_info[agent_id]['gt_goal'])
						info_rollout['gt_close'].append(agent_info[agent_id]['gt_close'])
						info_rollout['mask_nodes'].append(agent_info[agent_id]['mask_nodes'])

				if logging > 1:
					info_rollout['step_info'].append((node_id, edges_char))
					info_rollout['script'].append(agent_actions[agent_id])
					info_rollout['goals_finished'].append(env_info['satisfied_goals'])
					info_rollout['finished'] = env_info['finished']

					for agenti in range(len(self.agents)):
						info_rollout['action'][agenti].append(agent_actions[agenti])
						info_rollout['obs'].append(agent_info[agenti]['obs'])

					info_rollout['action_tried'].append(agent_info[agent_id]['action_tried'])
					if 'predicate' in agent_info[agent_id]:
						info_rollout['predicate'].append(agent_info[agent_id]['predicate'])
					info_rollout['graph'].append(curr_graph)
					info_rollout['action_space_ids'].append(agent_info[agent_id]['action_space_ids'])
					info_rollout['visible_ids'].append(agent_info[agent_id]['visible_ids'])
					info_rollout['reward'].append(reward)

			nb_steps += 1
			curr_num_steps += 1
			diff_reward = reward - prev_reward
			prev_reward = reward
			reward_step += diff_reward
			if 'bad_predicate' in agent_info[agent_id]:
				reward_step -= 0.2

			for agent_index in agent_info.keys():
				# currently single reward for both agents
				c_r_all[agent_index] += diff_reward
			
			if record:
				actions.append(agent_actions)

			# append to memory
			if is_train:
				for agent_id in range(self.num_agents):
					if 'RL' == self.agents[agent_id].agent_type or \
							self.agents[agent_id].agent_type == 'RL_MCTS' and 'mcts_action' not in agent_info[agent_id]:
						init_step_agent_info[agent_id] = agent_info[agent_id]

					# If this is the end of the action
					if 'RL' == self.agents[agent_id].agent_type or \
						self.agents[agent_id].agent_type == 'RL_MCTS' and self.agents[agent_id].action_count == 0:
						agent_steps += 1
						state = init_step_agent_info[agent_id]['state_inputs']
						policy = [log_prob.data for log_prob in init_step_agent_info[agent_id]['probs']]
						action = agent_info[agent_id]['actions']
						rewards = reward_step
						for i in range(self.num_agents):
							entropy_action.append(
								-((init_step_agent_info[agent_id]['probs'][i] + 1e-9).log() * init_step_agent_info[agent_id]['probs'][i]).sum().item())

						observation_space.append(init_step_agent_info[agent_id]['num_objects'])
						action_space.append(init_step_agent_info[agent_id]['num_objects_action'])
						last_agent_info = init_step_agent_info

						rollout_agent[agent_id].append((self.env.task_goal[agent_id], state, policy, action,
														rewards, curr_num_steps, 1))
						prev_reward_step = 0
						reward_step = 0
						curr_num_steps = 0

		if not is_train:
			pbar.close()
		t_steps = time.time() - t2
		for agent_index in agent_info.keys():
			success_r_all[agent_index] = env_info['finished']

		info_rollout['success'] = success_r_all[0]
		info_rollout['nsteps'] = nb_steps
		info_rollout['epsilon'] = self.agents[agent_id].epsilon
		info_rollout['entropy'] = (entropy_action, entropy_object)
		info_rollout['observation_space'] = np.mean(observation_space)
		info_rollout['action_space'] = np.mean(action_space)
		info_rollout['t_reset'] = t_reset
		info_rollout['t_steps'] = t_steps

		for agent_index in agent_info.keys():
			success_r_all[agent_index] = env_info['finished']


		info_rollout['env_id'] = self.env.env_id
		info_rollout['goals'] = list(self.env.task_goal[0].keys())

		# Rollout max
		if is_train:
			while nb_steps < self.max_number_steps:
				nb_steps += 1
				for agent_id in range(self.num_agents):
					if 'RL' in self.agents[agent_id].agent_type:
						state = last_agent_info[agent_id]['state_inputs']
						if 'edges' in obs.keys():
							pdb.set_trace()
						policy = [log_prob.data for log_prob in last_agent_info[agent_id]['probs']]
						action = last_agent_info[agent_id]['actions']
						# rewards = reward
						rollout_agent[agent_id].append((self.env.task_goal[agent_id], state, policy, action, 0, 0, 0))

		return c_r_all, info_rollout, rollout_agent


	def step(self, true_graph=False):
		if self.env.steps == 0:
			pass
		self.wait_if_paused()
		obs = self.env.get_observations()
		action_space = self.env.get_action_space()
		dict_actions, dict_info = self.get_actions(obs, action_space, true_graph=true_graph)
		for i in range(len(dict_info)):
			if len(dict_info) > 1 and 'subgoals' in dict_info[i]:
				dup = self.env.check_subgoal(dict_info[i]['subgoals'])
				self.cnt_nouse_subgoal += dup
				if i == 0 and 'subgoals' in dict_info[i + 1].keys() and dict_info[i]['subgoals'] == dict_info[i + 1]['subgoals']:
					self.cnt_duplicate_subgoal += 1
		try:
			step_info = self.env.step(dict_actions)
			if self.type == 'drama':
				self._update_obj_info(dict_actions)
			if self.type != 'mcts':
				time.sleep(3)  # to avoid too fast step
		except Exception as e:
			logger.info("Exception occurs when performing action: ", dict_actions)
			raise Exception
		return step_info, dict_actions, dict_info

	def run(self, random_goal=False, pred_goal=None, cnt_subgoal_info = False):
		"""
		self.task_goal: goal inference
		self.env.task_goal: ground-truth goal
		"""
		self.task_goal = copy.deepcopy(self.env.task_goal)
		agent_ids = [i for i, agent in self.agents.items() if agent.status != 'disconnected']
		if self.type == 'drama':
			self.heartbeat_manager = HeartbeatManager_run(agent_ids, 
												get_targets=lambda s: self._heartbeat_pairs_from_role(s), 
												on_disconnect=self.handle_agent_dropout,
												on_join=self.handle_agent_join)
		if random_goal:
			for predicate in self.env.task_goal[0]:
				u = random.choice([0, 1, 2])
				for i in range(self.num_agents):
					self.task_goal[i][predicate] = u
 
		if pred_goal is not None:
			self.task_goal = copy.deepcopy(pred_goal)

		no_success_time = 0

		init_dict = {i: [] for i in range(self.num_agents)}
		self.saved_info = {'task_id': self.env.task_id,
					  'env_id': self.env.env_id,
					  'task_name': self.env.task_name,
					  'gt_goals': self.env.task_goal[0],
					  'goals': self.task_goal,
					  'action': init_dict,
					  'plan': init_dict,
					  'subgoals': init_dict,
					  'finished': None,
					  'init_unity_graph': self.env.init_graph,
					  'goals_finished': [],
					  'belief': init_dict,
					  'belief_graph': init_dict,
					  'obs': init_dict,
					  'LLM': init_dict,
					  'graph': init_dict,
					  'progress': [],
					  'input_usage': 0,
					  'output_usage': 0
					}
		success = False
		end_putback = False
		rnd_agent = -1
		total_targets_num = sum(int(v) for v in (self.env.task_goal[0].values()))
		while True:
			if self.type == 'drama':
				if self.env.steps == 0:
					self.init_run = True
					self.pause()
			if (self.disconnected or self.turnover) and self.satisfied_disconnect_done:
				self.satisfied_disconnect_done = False
				if self.type == 'mcts':
					self.agents[rnd_agent].status = 'disconnected'
				elif self.type != 'drama':
					self.disconnect_agent(rnd_agent)
				else:
					self.heartbeat_manager.stop_flags[rnd_agent] = True
					self.heartbeat_manager.threads[rnd_agent].join()
					del self.heartbeat_manager.threads[rnd_agent]
					logger.info(f"Agent_{rnd_agent+1} heartbeat thread stopped and removed")
					self.pause()
				self.disconnect_count -= 1
			if (self.new_join or self.turnover) and self.satisfied_newjoin_done:
				self.satisfied_newjoin_done = False
				if self.turnover and self.turnover_same:
					self.agents[rnd_agent].status = 'active'
					logger.info(f"Agent_{rnd_agent + 1} re-joined the environment")
				else:
					if self.type != 'drama':
						self.join_agent(self.num_agents)
					else:
						logger.info(f"New agent Agent_{self.num_agents + 1} is going to be added")
						self.pause()
						self.heartbeat_manager.register_agent(self.num_agents)
				self.new_join_count -= 1


			(obs, reward, done, infos, messages), actions, agent_info = self.step()
			satisfied = {}
			for k, v in infos['progress']['satisfied'].items():
				count = sum(x is not None for x in v)
				if count > 0:
					satisfied[k] = count
			logger.info(f"satisfied: {satisfied}")

			if self.disconnected and not self.satisfied_disconnect_done:
				total_targets_num = sum(int(v) for v in (self.env.task_goal[0].values()))
				satisfied_count = sum(satisfied.values())
				ratio = (satisfied_count / total_targets_num) if total_targets_num > 0 else 0.0
				eligible_agents = [i for i, ag in self.agents.items() if ag.status != "disconnected"]
				if self.double_disconnect:
					if self.disconnect_count == 2 and ratio >= 0.3 and eligible_agents:
						random.seed(int(time.time() * 100))
						rnd_agent = random.choice(eligible_agents)
						logger.info(f"[30%] Schedule disconnect Agent_{rnd_agent + 1}")
						self.satisfied_disconnect_done = True
					if self.disconnect_count == 1 and ratio >= 0.6 and eligible_agents:
						random.seed(int(time.time() * 100))
						rnd_agent = random.choice(eligible_agents)
						logger.info(f"[60%] Schedule disconnect Agent_{rnd_agent + 1}")
						self.satisfied_disconnect_done = True
				else:
					if self.disconnect_count == 1 and ratio >= 0.3 and eligible_agents:
						random.seed(int(time.time() * 100))
						rnd_agent = random.choice(eligible_agents)
						logger.info(f"[30%] Schedule disconnect Agent_{rnd_agent + 1}")
						self.satisfied_disconnect_done = True

			if self.new_join and not self.satisfied_newjoin_done:
				total_targets_num = sum(int(v) for v in (self.env.task_goal[0].values()))
				satisfied_count = sum(satisfied.values())
				ratio = (satisfied_count / total_targets_num) if total_targets_num > 0 else 0.0

				if self.double_join:
					if self.new_join_count == 2 and ratio >= 0.3:
						rnd_step = self.env.steps + 1
						logger.info(f"[30%] Schedule new agent join at step {rnd_step}")
						self.satisfied_newjoin_done = True
					if self.new_join_count == 1 and ratio >= 0.6:
						rnd_step = self.env.steps + 1
						logger.info(f"[60%] Schedule new agent join at step {rnd_step}")
						self.satisfied_newjoin_done = True
				else:
					if self.new_join_count == 1 and ratio >= 0.3:
						rnd_step = self.env.steps + 1
						logger.info(f"[30%] Schedule new agent join at step {rnd_step}")
						self.satisfied_newjoin_done = True

			if self.turnover and not self.satisfied_disconnect_done and not self.satisfied_newjoin_done:
				total_targets_num = sum(int(v) for v in (self.env.task_goal[0].values()))
				satisfied_count = sum(satisfied.values())
				ratio = (satisfied_count / total_targets_num) if total_targets_num > 0 else 0.0
				eligible_agents = [i for i, ag in self.agents.items() if ag.status != "disconnected"]
				if self.disconnect_count == 1 and ratio >= 0.3:
					random.seed(int(time.time() * 100))
					rnd_agent = random.choice(eligible_agents)
					logger.info(f"[30%] Schedule turnover: disconnect Agent_{rnd_agent + 1}")
					self.satisfied_disconnect_done = True

				if self.new_join_count == 1 and ratio >= 0.6:
					rnd_step = self.env.steps + 1
					logger.info(f"[[60%] Schedule turnover: add agent at step {rnd_step}")
					self.satisfied_newjoin_done = True



			if end_putback and self.prev_satisfied == satisfied:
				if self.type == 'drama':
					self.assign_tasks_from_affinity()
				end_putback = False
			if self.prev_satisfied != satisfied and self.type == 'drama':
				self.prev_satisfied = satisfied
				end_putback = True
			
			if self.prev_satisfied != satisfied and self.type == 'agentverse':
				self.prev_satisfied = satisfied
				self.evaluate = True
				self.pause()

			success = infos['finished']
			if self.fail:
				done = True
				success = False
			if infos['failed_exec']:
				no_success_time += 1
				if no_success_time >= 4 or self.type == 'mcts':
					done = True
				self.reexecute = True
			else:
				self.reexecute = False
			if 'satisfied_goals' in infos:
				self.saved_info['goals_finished'].append(infos['satisfied_goals'])
			for agent_id, action in actions.items():
				self.saved_info['action'][agent_id].append(action)
			
			if 'progress' in infos:
				self.saved_info['progress'].append(infos['progress'])
			for agent_id, info in agent_info.items():
				if 'belief_graph' in info:
					self.saved_info['belief_graph'][agent_id].append(info['belief_graph'])
				if 'belief' in info:
					self.saved_info['belief'][agent_id].append(info['belief'])
				if 'plan' in info:
					self.saved_info['plan'][agent_id].append(info['plan'])
				if 'subgoals' in info:
					self.saved_info['subgoals'][agent_id].append(info['subgoals'])
				if 'obs' in info:
					self.saved_info['obs'][agent_id].append(copy.deepcopy(info['obs']))
				if 'LLM' in info:
					self.saved_info['LLM'][agent_id].append(info['LLM'])
				if 'graph' in info:
					self.saved_info['graph'][agent_id].append(copy.deepcopy(info['graph']))
				if 'input_usage' in info:
					self.saved_info['input_usage']+= info['input_usage']
				if 'output_usage' in info:
					self.saved_info['output_usage']+= info['output_usage']
				if self.debug:
					pickle.dump(self.saved_info, open(os.path.join(self.record_dir, 'log.pik'), 'wb'))
			if done:
				break
		self.saved_info['finished'] = success
		total_targets_num = sum(int(v) for v in (self.env.task_goal[0].values()))
		if cnt_subgoal_info:
			self.saved_info['cnt_duplicate_subgoal'] = self.cnt_duplicate_subgoal
			self.saved_info['cnt_nouse_subgoal'] = self.cnt_nouse_subgoal
			return success, self.env.steps, self.saved_info
		else:
			return success, self.env.steps, self.saved_info, total_targets_num

	def disconnect_agent(self, agent_id):
		logger.info(f"Remove agent_{agent_id + 1} from the environment")
		self.agents[agent_id].status = 'disconnected'
		disconnected_agent = self.agents[agent_id]
		disconnected_agent_room = disconnected_agent.current_room['class_name']
		obs = self.env.get_observations()
		if self.task_goal is None:
			goal_spec = self.env.get_goal(self.env.task_goal[agent_id], self.env.agent_goals[agent_id])
		else:
			goal_spec = self.env.get_goal(self.task_goal[agent_id], self.env.agent_goals[agent_id])
		disconnected_agent.obs_processing(obs[agent_id], goal_spec)
		grabbed_objects = []
		dropped_object_info = []
		disconnected_agent_name = f"Agent_{agent_id + 1}"

		if disconnected_agent.grabbed_objects is not None:
			# self.fail = True
			# return 
			grabbed_objects = disconnected_agent.grabbed_objects
			for obj in grabbed_objects:
				if obj in disconnected_agent.id2node:
					obj_node = disconnected_agent.id2node[obj]

					dropped_object_info.append({
						'id': obj_node['id'],
						'name': obj_node['class_name'],
						'room': disconnected_agent_room,
					})
					drop_script = f"<char{agent_id}> [putback] <{obj_node['class_name']}> ({obj}) <{disconnected_agent.current_room['class_name']}> ({disconnected_agent.current_room['id']})"
					try:
						success, message = self.env.comm.render_script([drop_script], 
															recording=False,
															skip_animation=True)
						if success:
							logger.info(f"Agent {agent_id + 1} disconnected. Object {obj} ({obj_node['class_name']}) dropped in {disconnected_agent.current_room['class_name']}")
						else:
							logger.info(f"Failed to drop object: {message}")
					except Exception as e:
						logger.info(f"Error dropping object: {e}")

	def join_agent(self, new_agent_id):
		self.agents[self.num_agents].status = 'active'

		self.saved_info['action'][new_agent_id] = []
		self.saved_info['plan'][new_agent_id] = []
		self.saved_info['subgoals'][new_agent_id] = []
		self.saved_info['belief'][new_agent_id] = []
		self.saved_info['belief_graph'][new_agent_id] = []
		self.saved_info['obs'][new_agent_id] = []
		self.saved_info['LLM'][new_agent_id] = []
		self.saved_info['graph'][new_agent_id] = []

		self.num_agents += 1
		self.env.num_agents = self.num_agents
	
	def _room_center(self):
		g = self.env.get_graph()
		centers = {}
		for n in g["nodes"]:
			if n.get("category") == "Rooms":
				name = n.get("class_name", "")
				obj_trans = n.get("obj_transform", "")
				pos = obj_trans.get("position")
				if isinstance(pos, dict):
					x, z = pos.get("x"), pos.get("z")
					if x is not None and z is not None:
						pos_xz = np.array([float(x), float(z)], dtype=np.float32)
				elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
					pos_xz = np.array([float(pos[0]), float(pos[2])], dtype=np.float32)

				if name and pos_xz is not None:
					centers[name] = pos_xz
		return centers

	def _agent_room_name(self, agent_idx: int) -> str:

		try:
			room = getattr(self.agents[agent_idx], "current_room", None)
			return room.get("class_name", "") if isinstance(room, dict) else ""
		except Exception:
			return ""

	def _room_center_distance(self, room_a: str, room_b: str, centers: dict) -> float:
		if not room_a or not room_b:
			return 1.0
		if room_a == room_b:
			return 0.0
		pa = centers.get(room_a)
		pb = centers.get(room_b)
		if pa is None or pb is None:
			return 1.0
		return float(np.linalg.norm(pa - pb))

	def _init_obj_info(self):
		rooms = list(getattr(self.env, "all_room_name", []))
		nrooms = len(rooms)
		task_info = defaultdict(lambda: {"count": 0, "destination": None, "prob": []})

		self._maybe_init_or_load_prob_net(rooms)

		for key, cnt in self.env.task_goal[0].items():
			parts = key.rsplit('_', 1)
			if len(parts) != 2 or not parts[1].isdigit():
				continue
			rel_obj = parts[0]                  # 'on_pudding'
			obj1 = rel_obj.split('_', 1)[-1]    # 'pudding'
			obj2 = int(parts[1])                # 268
			task_info[obj1]["count"] = int(cnt)
			task_info[obj1]["destination"] = obj2
			if not task_info[obj1]["prob"]:
				if obj1 in self.global_obj_room_probs:
					glob = self.global_obj_room_probs[obj1]
					total = sum(float(glob.get(r, 0.0)) for r in rooms)
					task_info[obj1]["prob"] = (
						{r: float(glob.get(r, 0.0)) / total for r in rooms} if total > 0
						else {r: 1.0 / max(1, nrooms) for r in rooms}
					)
				else:
					task_info[obj1]["prob"] = {r: 1.0 / max(1, nrooms) for r in rooms}
		self.task_info = dict(task_info)

	def _update_obj_info(self, dict_actions):
		for agent_id, action_info in dict_actions.items():
			if action_info == "None" or action_info is None:
				continue
			m_act = re.search(r"\[([^\]]+)\]", action_info)
			action = m_act.group(1) if m_act else ""
			m_obj = re.findall(r"<\s*([^>]+?)\s*>", action_info)
			obj = [m.strip() for m in m_obj]
			# logger.info(f"Agent_{agent_id + 1} action: {action}, object: {obj}")
			if action in {"grab"} and self.enable_prob_net:
				room = self._agent_room_name(agent_id)
				for ob in obj:
					if ob in self.task_info:
						prob_map = self.task_info[ob]["prob"]
						if isinstance(prob_map, dict) and prob_map and self.rooms_order:
							rooms = list(self.rooms_order)
							n = len(rooms)
							# 构造输入向量：旧分布 + 观测one-hot
							old = np.array([float(prob_map.get(r, 0.0)) for r in rooms], dtype=np.float32)
							obs = np.zeros(n, dtype=np.float32)
							if room in rooms:
								obs[rooms.index(room)] = 1.0
							inp = torch.from_numpy(np.concatenate([old, obs], axis=0)).unsqueeze(0)  # (1, 2n)
							# 前向获取新分布（softmax）
							with torch.no_grad():
								logits = self.prob_net(inp)  # (1, n)
								out = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
							# 融合，保证稳定
							eta = 0.9
							new = (1.0 - eta) * old + eta * out
							s = float(new.sum())
							if s > 0:
								new = new / s
							for r, v in zip(rooms, new.tolist()):
								prob_map[r] = float(v)
							# 可选：在线继续训练一步（用观测房间作标签）
							if self.prob_train and (room in rooms):
								self.prob_net.train()
								self.prob_opt.zero_grad(set_to_none=True)
								logits = self.prob_net(inp)  # (1, n)
								target = torch.tensor([rooms.index(room)], dtype=torch.long)
								loss = torch.nn.functional.cross_entropy(logits, target)
								loss.backward()
								self.prob_opt.step()
								self.prob_net.eval()
							# 更新全局缓存（稍后统一保存到磁盘）
							logger.info(f"[MLP] ob={ob} room={room} old={old.round(3)} out={out.round(3)} new={new.round(3)}")
							self.global_obj_room_probs[ob] = {r: float(prob_map[r]) for r in rooms}

			if action in {"putback", "putin"}:
				for ob in obj:
					if ob in self.task_info:
						self.task_info[ob]["count"] = max(0, self.task_info[ob]["count"] - 1)

	def _heartbeat_pairs_from_role(self, sender: int):
		pairs = []
		try:
			rows = getattr(self.agents[sender], "guardian_role_table", []) or []
			for row in rows:
				label = row.get("task")
				name = row.get("be_guarded", "none")
				if not label or not isinstance(name, str) or not name.startswith("Agent_"):
					continue
				try:
					idx = int(name.split("_", 1)[1]) - 1
					if idx != sender and idx >= 0:
						pairs.append((label, idx))
				except Exception:
					continue
		except Exception:
			pass
		return pairs

	def build_affinity_matrix(self):

		nA, nT = self.num_agents, len(self.task_info)
		A = np.zeros((nA, nT), dtype=np.float32)

		# 权重可调

		W_DIST = 0.5
		W_HANDS_FREE = 0.5

		centers = self._room_center()

		for i in range(nA):
			aroom = self._agent_room_name(i)
			# 是否空手（粗略）：若 agent.grabbed_objects 为空或 None
			objs = getattr(self.agents[i], "grabbed_objects", None) or []
			hands_free = 5.0 if len(objs) <= 1 else 0.0
			for j, task in enumerate(self.task_info.keys()):
				g = self.env.get_graph()
				for n in g["nodes"]:
					if n.get("id") == self.task_info[task]["destination"]:
						pos_destination = n.get("obj_transform").get("position")
						pos_xz = np.array([float(pos_destination[0]), float(pos_destination[2])], dtype=np.float32)
						break
				dist = 0.0
				prob_map = self.task_info[task]["prob"]
				if isinstance(prob_map, dict) and prob_map:
					for room, p in prob_map.items():
						if room in centers:
							dist += p * (self._room_center_distance(aroom, room, centers) + float(np.linalg.norm(pos_xz - centers[room])))
							continue
								
				score = 0.0
				
				score -= W_DIST * float(dist)
				score += W_HANDS_FREE * hands_free
				A[i, j] = score
		return A

	def _rebuild_guardian_chains_from_subgoals(self, A: np.ndarray, tasks: list):
		try:
			nA = len(self.agents)
			self.guardian_chain_units = {}
			# 选择每个 label 的 executor（谁拿到该 label 就是 executor）
			for ai in range(nA):
				for label in (self.subgoals_units.get(ai) or []):
					t = label.split('#', 1)[0]
					if t not in tasks:
						continue
					j = tasks.index(t)
					# 基于任务亲和度，executor 在首位，其余按 A[:, j] 递减
					base_order = list(np.argsort(-A[:, j]))
					order = [ai] + [a for a in base_order if a != ai]
					self.guardian_chain_units[label] = order

			# guardian_role_table
			guardian_table = {i: [] for i in range(nA)}
			for label, chain in self.guardian_chain_units.items():
				for pos, ag in enumerate(chain):
					role = "executor" if pos == 0 else f"guardian_{pos}"
					guard_name = "none" if pos == 0 else f"Agent_{chain[pos - 1] + 1}"
					be_guarded_name = f"Agent_{chain[pos + 1] + 1}" if pos + 1 < len(chain) else "none"
					guardian_table[ag].append({
						"task": label,
						"role": role,
						"guard": guard_name,
						"be_guarded": be_guarded_name
					})
			for i in range(nA):
				try:
					self.agents[i].guardian_role_table = guardian_table.get(i, [])
				except Exception:
					pass
			# 更新心跳路由
			if hasattr(self, "heartbeat_manager") and self.heartbeat_manager is not None:
				self.heartbeat_manager.set_target_resolver(lambda s: self._heartbeat_pairs_from_role(s))
		except Exception:
			pass

	def assign_tasks_from_affinity(self):
		A = self.build_affinity_matrix()
		nA, nT = A.shape
		if nT == 0:
			logger.info("No tasks to assign.")
			return

		# 1) 容量/需求/任务集
		def agent_capacity(i: int) -> int:
			objs = getattr(self.agents[i], "grabbed_objects", None) or []
			return max(0, 2 - len(objs))

		caps = [agent_capacity(i) for i in range(nA)]
		tasks = list(self.task_info.keys())

		# 构建 label 池：1..total（基于总目标数），并在分配时优先跳过“已完成数量”的编号
		if not hasattr(self, "task_labels"):
			self.task_labels = {}
		
		def _idx(lbl: str) -> int:
			try:
				return int(lbl.split('#', 1)[1])
			except Exception:
				return 10**9
		
		if not hasattr(self, "task_total_counts"):
			self.task_total_counts = {t: int(self.task_info[t]["count"]) for t in tasks}
		
		for t in tasks:
			try:
				total = int(self.task_total_counts.get(t, int(self.task_info[t]["count"])))
			except Exception:
				total = int(self.task_info[t].get("count", 0))
			total = max(1, total)  # 至少生成一个 label 以避免空集
			self.task_labels[t] = [f"{t}#{k}" for k in range(1, total + 1)]

		assigned_labels = set()

		def _label_index(lb: str) -> int:
			try:
				return int(lb.split('#', 1)[1])
			except Exception:
				return 0

		def alloc_label(t: str) -> str:
			# 根据“已完成数量 completed = total - current_remaining”确定起始编号
			total = int(self.task_total_counts.get(t, max(0, int(self.task_info.get(t, {}).get("count", 0)))))
			current = int(self.task_info.get(t, {}).get("count", 0))
			completed = max(0, total - current)

			# 选择 idx > completed 的最小未占用编号
			candidates = []
			for lb in self.task_labels.get(t, []):
				if lb in assigned_labels:
					continue
				idx = _label_index(lb)
				if idx > completed:
					candidates.append((idx, lb))
			if candidates:
				candidates.sort()
				chosen = candidates[0][1]
				assigned_labels.add(chosen)
				return chosen

			# 回退：任意未占用的最小编号
			for lb in sorted(self.task_labels.get(t, []), key=_idx):
				if lb not in assigned_labels:
					assigned_labels.add(lb)
					return lb

			# 兜底：扩容一个新编号
			new_idx = max((_label_index(lb) for lb in self.task_labels.get(t, [])), default=0) + 1
			new_lb = f"{t}#{new_idx}"
			self.task_labels.setdefault(t, []).append(new_lb)
			assigned_labels.add(new_lb)
			logger.warning(f"[assign] No free label for task {t}, created {new_lb}")
			return new_lb

		try:
			g = self.env.get_graph()
			id2node = {n["id"]: n for n in g["nodes"]}
		except Exception:
			id2node = {}

		held_counts = {i: defaultdict(int) for i in range(nA)}
		for i in range(nA):
			held = getattr(self.agents[i], "grabbed_objects", None) or []
			for oid in held:
				node = id2node.get(oid)
				if node:
					held_counts[i][node.get("class_name", "").lower()] += 1

		demands = [max(0, int(self.task_info[t]["count"])) for t in tasks]
		logger.info(f"Task demands: {demands}")

		# ---------- 预分配（持有物优先） ----------
		pre_assign = {i: [] for i in range(nA)}
		pre_assign_units = {i: [] for i in range(nA)}
		for i in range(nA):
			if caps[i] <= 0:
				continue
			for j, t in enumerate(tasks):
				if caps[i] <= 0:
					break
				need = demands[j]
				if need <= 0:
					continue
				k = held_counts[i].get(t.lower(), 0)  # 手里拿了多少该物体
				if k <= 0:
					continue
				assign_k = min(k, need, caps[i])
				if assign_k <= 0:
					continue
				pre_assign[i].extend([t] * assign_k)
				for _ in range(assign_k):
					label = alloc_label(t)
					if label is None:
						continue
					pre_assign_units[i].append(label)
				demands[j] -= assign_k
				caps[i] -= assign_k

		logger.info(f"Task demands(after pre-assign): {demands}")

		total_cap = sum(caps)
		total_dem = sum(demands)

		# 若预分配已满足全部需求：直接写结果
		if total_dem == 0:
			self.subgoals = {}
			self.subgoals_units = {}
			for i in range(nA):
				names = pre_assign[i]
				units = pre_assign_units[i]
				self.subgoals[i] = names if names else None
				self.subgoals_units[i] = units if units else None
			# guardian 链（仅基于亲和度与执行者）
			self._rebuild_guardian_chains_from_subgoals(A, tasks)
			return

		# 无剩余容量：仅保留预分配
		if total_cap == 0:
			self.subgoals = {}
			self.subgoals_units = {}
			for i in range(nA):
				names = pre_assign[i]
				units = pre_assign_units[i]
				self.subgoals[i] = names if names else None
				self.subgoals_units[i] = units if units else None
			logger.info(f"No capacity for matching. total_cap={total_cap}")
			self._rebuild_guardian_chains_from_subgoals(A, tasks)
			return

		# ---------- 匹配分配（Hungarian） ----------
		row_slots = []
		for i in range(nA):
			row_slots.extend([i] * caps[i])
		col_slots = []
		for j in range(nT):
			col_slots.extend([j] * demands[j])

		m, n = len(row_slots), len(col_slots)
		if n == 0 or m == 0:
			self.subgoals = {}
			self.subgoals_units = {}
			for i in range(nA):
				names = pre_assign[i]
				units = pre_assign_units[i]
				self.subgoals[i] = names if names else None
				self.subgoals_units[i] = units if units else None
			logger.info(f"Skip matching. m={m}, n={n}")
			self._rebuild_guardian_chains_from_subgoals(A, tasks)
			return

		A_exp = np.zeros((m, n), dtype=np.float32)
		for ri, ai in enumerate(row_slots):
			A_exp[ri, :] = A[ai, col_slots]

		pad_mn = max(m, n)
		BIG_M = 1e6
		pad_cost = np.full((pad_mn, pad_mn), BIG_M, dtype=np.float32)
		pad_cost[:m, :n] = -A_exp
		row_ind, col_ind = linear_sum_assignment(pad_cost)
		pairs = [(r, c) for r, c in zip(row_ind, col_ind) if r < m and c < n]

		assign_map = {i: [] for i in range(nA)}
		assign_map_units = {i: [] for i in range(nA)}
		assigned_counts = [0] * nT

		for r, c in pairs:
			ai = row_slots[r]
			tj = col_slots[c]
			if assigned_counts[tj] >= demands[tj]:
				continue
			t = tasks[tj]
			assigned_counts[tj] += 1
			assign_map[ai].append(t)
			label = alloc_label(t)
			if label is None:
				continue
			assign_map_units[ai].append(label)

		# 合并“预分配 + 匹配分配”
		self.subgoals = {}
		self.subgoals_units = {}
		for i in range(nA):
			names = pre_assign[i] + assign_map[i]
			units = pre_assign_units[i] + assign_map_units[i]
			self.subgoals[i] = names if names else None
			self.subgoals_units[i] = units if units else None

		# guardian 链（仅基于亲和度与执行者）
		self._rebuild_guardian_chains_from_subgoals(A, tasks)

		# 日志
		try:
			mat_str = np.array2string(A, precision=2, floatmode="fixed")
			logger.info(f"Affinity matrix A (agents x tasks):\n{mat_str}")
			pretty = []
			for i in range(nA):
				pretty.append(f"Agent_{i+1} -> {self.subgoals_units.get(i)}")
			logger.info("Assignment(units): " + " | ".join(pretty))
			logger.info("Guardian chain (per unit):")
			def _label_key(lbl: str):
				t, idx = lbl.split('#', 1)
				try:
					return (tasks.index(t), int(idx))
				except Exception:
					return (0, 0)
			for label in sorted(self.guardian_chain_units.keys(), key=_label_key):
				t = label.split('#', 1)[0]
				j = tasks.index(t)
				order = self.guardian_chain_units[label]
				chain = " > ".join([f"Agent_{ai+1}:{A[ai, j]:.2f}" for ai in order])
				logger.info(f"  {label}: {chain}")
		except Exception:
			pass

	def _drop_objects_in_place(self, agent_idx: int):
		try:
			disconnected_agent = self.agents[agent_idx]
			disconnected_agent_room = disconnected_agent.current_room['class_name']
			grabbed_objects = []
			dropped_object_info = []
			if disconnected_agent.grabbed_objects is not None:
				grabbed_objects = disconnected_agent.grabbed_objects

				for obj_id in grabbed_objects:
					if obj_id in disconnected_agent.id2node:
						obj_node = disconnected_agent.id2node[obj_id]

						dropped_object_info.append({
							'id': obj_id,
							'name': obj_node['class_name'],
							'room':disconnected_agent_room
						})

						# TODO: drop the object in the current room
						drop_script = f"<char{agent_idx}> [PutBack] <{obj_node['class_name']}> ({obj_id}) <{disconnected_agent.current_room['class_name']}> ({disconnected_agent.current_room['id']})"
						try:
							success, message = self.env.comm.render_script([drop_script], 
																   recording=False,
																   skip_animation=True)
							if success:
								logger.info(f"Agent {agent_idx + 1} disconnected. Object {obj_id} ({obj_node['class_name']}) dropped in {disconnected_agent.current_room['class_name']}")
							else:
								logger.info(f"Failed to drop object: {message}")
						except Exception as e:
							logger.info(f"Error dropping object: {e}")

				self.env.changed_graph = True
		except Exception as e:
			logger.info(f"_drop_objects_in_place error: {e}")

	def handle_agent_dropout(self, dead_agents: List[int]):
		if not dead_agents:
			return		
		self.pause()
		for d in dead_agents:
			self.agents[d].status = 'disconnected'
			self._drop_objects_in_place(d)
		try:
			logger.info(f"Handling dropout for agents: {[f'Agent_{d+1}' for d in dead_agents]}")
			# 1) adjust guardian chains
			for label, chain in list(self.guardian_chain_units.items()):
				orig_chain = list(chain)
				# any dead in chain?
				if not any(d in orig_chain for d in dead_agents):
					continue
				# remove all dead agents preserving order
				new_chain = [a for a in orig_chain if a not in dead_agents]
				if not new_chain:
					# nobody left guarding this unit
					logger.info(f"  {label}: all guardians dropped -> remove chain")
					del self.guardian_chain_units[label]
					# remove from any subgoals_units
					for ai in range(len(self.agents)):
						try:
							if label in (self.subgoals_units.get(ai) or []):
								self.subgoals_units[ai].remove(label)
						except Exception:
							pass
					continue
				# if executor died, promote new executor (new_chain[0])
				executor_died = orig_chain[0] in dead_agents
				old_executor = orig_chain[0]
				self.guardian_chain_units[label] = new_chain
				if executor_died:
					new_executor = new_chain[0]
					# ensure label assigned to new_executor
					# remove label from dead executors' subgoals_units
					for d in dead_agents:
						try:
							if label in (self.subgoals_units.get(d) or []):
								self.subgoals_units[d].remove(label)
						except Exception:
							pass
					# add to new executor if not present
					try:
						if label not in (self.subgoals_units.get(new_executor) or []):
							self.subgoals_units.setdefault(new_executor, []).append(label)
					except Exception:
						pass
					# also update self.subgoals (task names) accordingly
					try:
						t = label.split('#', 1)[0]
						if self.subgoals.get(new_executor) is None:
							self.subgoals[new_executor] = []
						if t not in self.subgoals[new_executor]:
							self.subgoals[new_executor].append(t)
					except Exception:
						pass
					logger.info(f"  {label}: executor Agent_{old_executor+1} dropped -> promoted Agent_{new_executor+1}")
				else:
					logger.info(f"  {label}: removed dead guardians {dead_agents}, new chain: {[f'Agent_{a+1}' for a in new_chain]}")

			# 2) rebuild guardian_role_table
			nA = len(self.agents)
			guardian_table = {i: [] for i in range(nA)}
			for label, chain in self.guardian_chain_units.items():
				for pos, ai in enumerate(chain):
					role = "executor" if pos == 0 else f"guardian_{pos}"
					guard_name = "none" if pos == 0 else f"Agent_{chain[pos - 1] + 1}"
					be_guarded_name = f"Agent_{chain[pos + 1] + 1}" if pos + 1 < len(chain) else "none"
					guardian_table[ai].append({
						"task": label,
						"role": role,
						"guard": guard_name,
						"be_guarded": be_guarded_name
					})
			# write back to agents
			for i in range(nA):
				try:
					self.agents[i].guardian_role_table = guardian_table.get(i, [])
				except Exception:
					pass

			# 3) refresh heartbeat resolver so threads will send according to updated be_guarded
			try:
				if hasattr(self, "heartbeat_manager") and self.heartbeat_manager is not None:
					self.heartbeat_manager.set_target_resolver(lambda s: self._heartbeat_pairs_from_role(s))
			except Exception as e:
				logger.info(f"Failed to update heartbeat resolver after dropout: {e}")

			logger.info("Dropout handling completed and guardian roles updated.")
		except Exception as e:
			logger.info(f"handle_agent_dropout error: {e}")

		finally:
			self.resume()

	def _next_label_index(self, existing_labels: List[str], task: str) -> int:
		mx = 0
		for lb in existing_labels:
			if not lb.startswith(task + "#"):
				continue
			try:
				k = int(lb.split('#', 1)[1])
				mx = max(mx, k)
			except Exception:
				continue
		return mx + 1
	
	def _score_agent_task_single(self, agent_idx: int, task_name: str) -> float:
		try:
			W_DIST = 0.5
			W_HANDS_FREE = 0.5
			centers = self._room_center()
			aroom = self._agent_room_name(agent_idx)
			objs = getattr(self.agents[agent_idx], "grabbed_objects", None) or []
			hands_free = 5.0 if len(objs) <= 1 else 0.0

			g = self.env.get_graph()
			pos_xz = None
			dest_id = self.task_info[task_name]["destination"]
			for n in g["nodes"]:
				if n.get("id") == dest_id:
					pos_destination = n.get("obj_transform").get("position")
					pos_xz = np.array([float(pos_destination[0]), float(pos_destination[2])], dtype=np.float32)
					break

			dist = 0.0
			prob_map = self.task_info[task_name]["prob"]
			if isinstance(prob_map, dict) and prob_map:
				for room, p in prob_map.items():
					if room in centers and pos_xz is not None:
						dist += p * (self._room_center_distance(aroom, room, centers) + float(np.linalg.norm(pos_xz - centers[room])))
						break

			score = 0.0
			score -= W_DIST * float(dist)
			score += W_HANDS_FREE * hands_free
			return float(score)
		except Exception:
			return 0.0

	def _executor_has_grabbed(self, agent_idx: int, task: str) -> bool:
		held = getattr(self.agents[agent_idx], "grabbed_objects", None) or []
		if not held:
			return False
		g = self.env.get_graph()
		id2node = {n["id"]: n for n in g["nodes"]}
		t = task.lower()
		for oid in held:
			node = id2node.get(oid)
			if node and node.get("class_name", "").lower() == t:
				return True
		return False

	def handle_agent_join(self, new_agent_id: int):
		self.pause()
		try:
			if new_agent_id in self.agents and self.agents[new_agent_id].status == 'disconnected' and new_agent_id != self.num_agents:
				self.agents[new_agent_id].status = 'active'
				logger.info(f"Agent_{new_agent_id + 1} re-joined the environment")
			elif new_agent_id == self.num_agents:
				self.join_agent(new_agent_id)
				logger.info(f"Agent_{new_agent_id + 1} joined the environment")

			tasks = list(self.task_info.keys())
			by_task = {}
			for label in self.guardian_chain_units.keys():
				t = label.split('#', 1)[0]
				if t not in tasks:
					continue
				by_task.setdefault(t, []).append(label)

			free_labels: List[Tuple[str, str]] = []  # (task, label)
			for t in tasks:
				total = int(self.task_info[t]["count"])
				assigned = len(by_task.get(t, []))
				if assigned < total:
					next_idx = self._next_label_index(by_task.get(t, []), t)
					for k in range(assigned + 1, total + 1):
						label = f"{t}#{next_idx + (k - assigned - 1)}"
						free_labels.append((t, label))

			caps = 2
			# 1) assign some free labels, or preempt from others if beneficial
			if free_labels:
				scores = [(self._score_agent_task_single(new_agent_id, t), t, label) for (t, label) in free_labels]
				scores.sort(reverse=True)
				take = min(len(scores), 2)
				picked = scores[:take]
				for _, t, label in picked:
					tmpl_chain = None
					for ex_label, chain in self.guardian_chain_units.items():
						if ex_label.split('#', 1)[0] == t:
							tmpl_chain = [a for a in chain if a != new_agent_id]
							break
					if tmpl_chain is None:
						tmpl_chain = [i for i in range(self.num_agents) if i != new_agent_id and self.agents[i].status == 'active']

					chain = [new_agent_id] + tmpl_chain
					self.guardian_chain_units[label] = chain
					self.subgoals_units.setdefault(new_agent_id, [])
					if label not in (self.subgoals_units[new_agent_id] or []):
						self.subgoals_units[new_agent_id].append(label)
					self.subgoals.setdefault(new_agent_id, [])
					if t not in (self.subgoals[new_agent_id] or []):
						self.subgoals[new_agent_id].append(t)
					caps -= 1
					logger.info(f"[JOIN] assign free label {label} -> Agent_{new_agent_id+1}")


			if caps > 0:
				candidates = []
				for label, chain in self.guardian_chain_units.items():
					if not chain:
						continue
					old_executor = chain[0]
					if old_executor == new_agent_id:
						continue
					t = label.split('#', 1)[0]

					if self._executor_has_grabbed(old_executor, t):
						# executor already grabbed the object, skip for now
						continue

					v_new = self._score_agent_task_single(new_agent_id, t)
					v_old = self._score_agent_task_single(old_executor, t)
					value = v_new - v_old
					candidates.append((value, t, label, old_executor))

				candidates.sort(reverse=True, key=lambda x: x[0])
				for value, t, label, old_executor in candidates:
					if caps <= 0:
						break
					if value <= 0:
						break

					old_chain = self.guardian_chain_units[label]
					rest = [a for a in old_chain if a != new_agent_id]
					new_order = [new_agent_id] + rest
					self.guardian_chain_units[label] = new_order

					if label in (self.subgoals_units.get(old_executor) or []):
						self.subgoals_units[old_executor].remove(label)
					t = label.split('#', 1)[0]
					still_has = any(lab.split('#', 1)[0] == t for lab in (self.subgoals_units.get(old_executor) or []))
					if not still_has and t in (self.subgoals.get(old_executor) or []):
						self.subgoals[old_executor].remove(t)

					self.subgoals_units.setdefault(new_agent_id, [])
					if label not in (self.subgoals_units[new_agent_id] or []):
						self.subgoals_units[new_agent_id].append(label)
					self.subgoals.setdefault(new_agent_id, [])
					if t not in (self.subgoals[new_agent_id] or []):
						self.subgoals[new_agent_id].append(t)

					caps -= 1
					logger.info(f"[JOIN] preempt label {label} from Agent_{old_executor+1} -> Agent_{new_agent_id+1}")
			# 2) rebuild guardian_role_table
			nA = len(self.agents)
			guardian_table = {i: [] for i in range(nA)}
			for label, chain in self.guardian_chain_units.items():
				for pos, ai in enumerate(chain):
					role = "executor" if pos == 0 else f"guardian_{pos}"
					guard_name = "none" if pos == 0 else f"Agent_{chain[pos - 1] + 1}"
					be_guarded_name = f"Agent_{chain[pos + 1] + 1}" if pos + 1 < len(chain) else "none"
					guardian_table[ai].append({
						"task": label,
						"role": role,
						"guard": guard_name,
						"be_guarded": be_guarded_name
					})
			# write back to agents
			for i in range(nA):
				try:
					self.agents[i].guardian_role_table = guardian_table.get(i, [])
				except Exception:
					pass
			try:
				if hasattr(self, "heartbeat_manager") and self.heartbeat_manager is not None:
					self.heartbeat_manager.set_target_resolver(lambda s: self._heartbeat_pairs_from_role(s))
			except Exception as e:
				logger.info(f"Failed to update heartbeat resolver after join: {e}")
		
		except Exception as e:
			logger.info(f"handle_agent_join error: {e}")
		finally:
			self.resume()
