import time
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
import logging
logger = logging.getLogger("__main__")


class HeartbeatManager:
	def __init__(self, 
			  agents: List[int], 
			  timeout: int = 10, 
			  get_targets: Optional[Callable[[int], List[Tuple[str, int]]]] = None, 
			  on_disconnect: Optional[Callable[[List[int]], None]] = None,
			  watchdog_interval: float = 1.0,
			  on_join: Optional[Callable[[int], None]] = None,
			  allow_autoregister: bool = True):
		self.agents = agents
		self.timeout = timeout
		self.watchdog_interval = watchdog_interval
		current_time = time.time()
		self.heartbeat_timestamps = {agent : current_time for agent in agents}
		self.received_heartbeat_label: Dict[int, Dict[str, Dict[int, float]]] = {
			agent: {} for agent in agents
		}
		self.lock = Lock()

		self.threads = {}
		self.stop_flags = {agent: False for agent in agents}
		self.dead_agents: Set[int] = set()

		self.hello_received = {}
		self.pending_agents = []

		self.get_targets: Optional[Callable[[int], List[Tuple[str, int]]]] = get_targets
		self.on_disconnect: Optional[Callable[[List[int]], None]] = on_disconnect
		self.on_join: Optional[Callable[[int], None]] = on_join
		self.allow_autoregister = allow_autoregister

		self._stop_watchdog = False
		self._watchdog_thread = Thread(target=self._watchdog_loop, daemon=True)
		self._watchdog_thread.start()

	def set_target_resolver(self, func: Callable[[int], List[Tuple[str, int]]]):
		self.get_targets = func

	def update_heartbeat(self, sender: int, receiver: int):
		with self.lock:
			current_time = time.time()
			if sender not in self.heartbeat_timestamps and self.allow_autoregister:
				pass_to_register = sender
			else:
				pass_to_register = None
			if sender in self.heartbeat_timestamps:
				self.heartbeat_timestamps[sender] = current_time
		if pass_to_register is not None and self.allow_autoregister:
			self.register_agent(pass_to_register)
	
	def _watchdog_loop(self):
		while not self._stop_watchdog:
			try:
				dead = self.check_disconnect_agent()
			except Exception as e:
				logger.info(f"Error in watchdog loop: {e}")
				dead = []
			if dead:
				logger.info(f"[HB] watchdog detected dropout: {[f'Agent_{d+1}' for d in dead]}")
				try:
					if self.on_disconnect:
						self.on_disconnect(dead)
				except Exception as e:
					logger.info(f"[HB] on_disconnect callback error: {e}")

			try:
				joined = self.check_new_agent()
			except Exception as e:
				logger.info(f"Error in check_new_agent: {e}")
				joined = []
			if joined:
				logger.info(f"[HB] watchdog detected new agents: {[f'Agent_{d+1}' for d in joined]}")
				for agent in joined:
					try:
						if self.on_join:
							self.on_join(agent)
					except Exception as e:
						logger.info(f"[HB] on_join callback error: {e}")

			time.sleep(self.watchdog_interval)

	def register_agent(self, agent_id: int) -> bool:
		with self.lock:
			if agent_id in self.agents:
				self.dead_agents.discard(agent_id)
				self.stop_flags[agent_id] = False
				if agent_id not in self.threads or not self.threads[agent_id].is_alive():
					th = Thread(target=self.broadcast_heartbeat, args=(agent_id, 1.0), daemon=True)
					th.start()
					self.threads[agent_id] = th
				logger.info(f"[HB] register_agent: Agent_{agent_id+1} already known -> ensured running")
				return False
		
			self.agents.append(agent_id)
			self.heartbeat_timestamps[agent_id] = time.time()
			self.received_heartbeat_label[agent_id] = {}
			self.stop_flags[agent_id] = False
			self.dead_agents.discard(agent_id)
			th = Thread(target=self.broadcast_heartbeat, args=(agent_id, 1.0), daemon=True)
			th.start()
			self.threads[agent_id] = th
			logger.info(f"[HB] register_agent: Agent_{agent_id+1} joined (threads started)")
		try:
			if self.on_join:
				self.on_join(agent_id)
		except Exception as e:
			logger.info(f"[HB] on_join callback error: {e}")
		return True

	
	def broadcast_heartbeat(self, sender, heartbeat_interval: float = 1.0):
		while not self.stop_flags.get(sender, False):
			try:
				current_time = time.time()
				with self.lock:
					self.heartbeat_timestamps[sender] = current_time
					# for receiver in self.agents:
					# 	if receiver != sender and receiver in self.received_heartbeat:
					# 		self.received_heartbeat[receiver][sender] = current_time
					pairs: List[Tuple[str, int]] = []
					if self.get_targets is not None:
						try:
							raw = self.get_targets(sender) or []
						except Exception as e:
							logger.info(f"get_targets error for Agent_{sender+1}: {e}")
							raw = []
						# 解析并过滤：只接受 (label:str, recv:int)
						for it in raw:
							if isinstance(it, (list, tuple)) and len(it) >= 2:
								try:
									label = str(it[0])
									recv = int(it[1])
								except Exception:
									continue
								if recv in self.agents and recv != sender:
									pairs.append((label, recv))

					for label, receiver in pairs:
						if receiver not in self.received_heartbeat_label:
							self.received_heartbeat_label[receiver] = {}
						if label not in self.received_heartbeat_label[receiver]:
							self.received_heartbeat_label[receiver][label] = {}
						self.received_heartbeat_label[receiver][label][sender] = current_time

					# 日志本轮发送（便于检查）
					if pairs:
						pretty = ", ".join([f"{label}->Agent_{recv+1}" for label, recv in pairs])
					else:
						pretty = "(none)"
					# logger.info(f"[HB] Agent_{sender+1} send: {pretty}")
				time.sleep(heartbeat_interval)
			except Exception as e:
				logger.info(f"Error in heartbeat thread for Agent_{sender}: {e}")
				time.sleep(heartbeat_interval * 2)
				break
	
	def broadcast_message(self, sender: int, receiver: int, message: str):
		if message == "hello" and sender in self.pending_agents:
			if receiver in self.agents:
				self.hello_received[sender][receiver] = True

	def check_disconnect_agent(self) -> List[int]:
		current_time = time.time()
		disconnected_agents: List[int] = []

		with self.lock:
			for agent in list(self.agents):
				last_heartbeat = self.heartbeat_timestamps.get(agent, 0.0)
				if agent in self.dead_agents:
					continue
				last_heartbeat = self.heartbeat_timestamps.get(agent, 0.0)
				if current_time - last_heartbeat > self.timeout:
					disconnected_agents.append(agent)

		if disconnected_agents:
			self.dead_agents.update(disconnected_agents)
			for dead in disconnected_agents:
				self.stop_flags[dead] = True
				for receiver in self.agents:
					if receiver == dead:
						continue
					labels = self.received_heartbeat_label.get(receiver, {})
					affected = []
					for label, senders in labels.items():
						if dead in senders:
							senders[dead] = 0.0
							affected.append(label)
					if affected:
						logger.info(f"Heartbeat dropout: Agent_{dead+1} -> propagated to Agent_{receiver+1} labels={affected}")
		return disconnected_agents

	def check_new_agent(self) -> List[int]:
		new_agents = []
		with self.lock:
			if self.pending_agents:
				for agent in list(self.pending_agents):
					if all(self.hello_received[agent].values()):
						self.agents.append(agent)
						heartbeat_thread = Thread(target=self.broadcast_heartbeat, args=(agent, 1.0))
						heartbeat_thread.daemon = True
						heartbeat_thread.start()
						self.threads[agent] = heartbeat_thread
						self.pending_agents.remove(agent)
						self.stop_flags[agent] = False
						self.dead_agents.discard(agent)
						self.heartbeat_timestamps[agent] = time.time()
						if agent not in self.received_heartbeat_label:
							self.received_heartbeat_label[agent] = {}
						new_agents.append(agent)

		return new_agents
	
	def stop_all_threads(self):
		for agent in self.agents:
			self.stop_flags[agent] = True
			if agent in self.threads and self.threads[agent].is_alive():
				try:
					self.threads[agent].join(timeout=1)
				except Exception as e:
					logger.info(f"Error stopping heartbeat thread for Agent_{agent}: {e}")
		self.threads.clear()
	
	
def HeartbeatManager_run(
	agents: List[int],
	get_targets: Optional[Callable[[int], List[Tuple[str, int]]]] = None,
	on_disconnect: Optional[Callable[[List[int]], None]] = None,
	on_join: Optional[Callable[[int], None]] = None,
	watchdog_interval: float = 0.5,
):
	hb = HeartbeatManager(agents, 
					   get_targets=get_targets, 
					   on_disconnect=on_disconnect,
					   on_join=on_join,
					   watchdog_interval=watchdog_interval)
	for agent in list(agents):
		th = Thread(target=hb.broadcast_heartbeat, args=(agent, 1.0), daemon=True)
		th.start()
		hb.threads[agent] = th
		time.sleep(0.05)
	return hb