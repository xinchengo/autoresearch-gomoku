from __future__ import annotations

import numpy as np
import multiprocessing as mp
from multiprocessing import connection


def _worker(remote: connection.Connection, parent_remote: connection.Connection, env_fn):
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                remote.send((obs, reward, terminated, truncated, info, env.board.copy()))
            elif cmd == "reset":
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise ValueError(f"unknown command: {cmd}")
    except (EOFError, KeyboardInterrupt):
        pass


class SubprocVecEnv:
    def __init__(self, env_fns: list):
        self.num_envs = len(env_fns)
        self.remotes: list[connection.Connection] = []
        self.workers: list[connection.Connection] = []
        self.processes: list[mp.Process] = []

        for env_fn in env_fns:
            worker_conn, parent_conn = mp.Pipe()
            self.remotes.append(parent_conn)
            self.workers.append(worker_conn)
            p = mp.Process(target=_worker, args=(worker_conn, parent_conn, env_fn))
            p.daemon = True
            p.start()
            worker_conn.close()
            self.processes.append(p)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, terminateds, truncateds, infos, boards = zip(*results)
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            list(infos),
            list(boards),
        )

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> tuple[np.ndarray, list]:
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)

    def close(self) -> None:
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join(timeout=2)
