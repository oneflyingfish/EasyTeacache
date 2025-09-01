import numpy as np
import torch
from typing import Tuple, List, Dict
from collections import deque


class TeaCacheConfig:
    def __init__(self, complexity=4, fit_length=0):
        self.complexity = complexity
        self.coefficients = None  # np.ndarray
        self.evaluate_func = None
        self.fit_length = fit_length
        self.calibration_size = -1

    def matching_rate(self, length) -> float:
        return abs(length - self.fit_length)

    def __deepcopy__(self, memo):
        import copy

        cls = self.__class__
        new_obj = cls.__new__(cls, self.complexity)  # type:TeaCacheConfig

        for name, value in self.__dict__.items():
            if name in ["evaluate_func"]:
                setattr(new_obj, name, None)
            else:
                setattr(new_obj, name, copy.deepcopy(value, memo))

        new_obj.update_func()
        return new_obj

    def set_coefficients(self, coefficients: tuple | np.ndarray) -> np.ndarray:
        if isinstance(coefficients, np.ndarray):
            self.coefficients = coefficients
        else:
            self.coefficients = np.array(coefficients, dtype=np.float32)
        return self.coefficients

    def set_coefficients_by_points(self, points: Tuple[np.ndarray]) -> np.ndarray:
        if len(points[0]) <= self.complexity:
            return None

        self.calibration_size = len(points[0])
        self.coefficients = np.polyfit(points[0], points[1], self.complexity)
        return self.coefficients

    def update_func(self, points: Tuple[np.ndarray] = None) -> bool:
        if points is not None:
            self.coefficients = self.set_coefficients_by_points(points)

        if self.coefficients is None:
            return False
        self.evaluate_func = np.poly1d(self.coefficients)
        return True

    def valid(self) -> bool:
        return self.evaluate_func is not None

    def evaluate(self, input) -> float:
        if self.evaluate_func is None:
            return None

        return self.evaluate_func(input)

    def from_json(self, data: dict) -> bool:
        if data is None or len(data) < 1:
            return False

        import copy

        for name, value in self.__dict__.items():
            if name not in ["evaluate_func"]:
                if name not in data:
                    print(f'warning: bad config, need "{name}"')
                    continue
                setattr(self, name, copy.deepcopy(data[name]))

        if (
            not isinstance(self.coefficients, np.ndarray)
            and self.coefficients is not None
        ):
            self.coefficients = np.array(self.coefficients, dtype=np.float32)
        self.update_func()
        return True

    def to_json(self) -> dict:
        data = {}
        for name, value in self.__dict__.items():
            if name not in ["evaluate_func"]:
                data[name] = (
                    value if not isinstance(value, np.ndarray) else value.tolist()
                )
        return data

    @staticmethod
    def remove_from_file(path: str, keys: List = None) -> bool:
        from pathlib import Path
        import os, json

        save_path = Path(path)
        os.makedirs(save_path.parent, exist_ok=True)

        config = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as fp:
                    config = json.load(fp)
            except Exception as ex:
                print(f"fail to analyse teacache config from {path}, skip:", str(ex))
                config = {}

        ptr = config
        if keys is not None and len(keys) > 0:
            for key in keys:
                if key in ptr:
                    ptr = ptr[key]
                else:
                    ptr[key] = {}
                    ptr = ptr[key]
        if "meta_data" in ptr:
            del ptr["meta_data"]

        try:
            with open(path, "w") as fp:
                json.dump(config, fp, indent=4)
        except Exception as ex:
            print("fail to save teacache config:", str(ex))

    def load_file(self, path: str, keys: List = None) -> bool:
        import os, json

        if not os.path.exists(path):
            self.coefficients = None
            return False

        config = {}
        with open(path, "r") as fp:
            config = dict(json.load(fp))

        if keys is not None:
            for key in keys:
                if key not in config:
                    self.coefficients = None
                    return False
                else:
                    config = config[key]
        return self.from_json(config.get("meta_data", None))

    @staticmethod
    def load_files(path: str, keys: List = None, only_valid=True) -> list | None:
        teacache_configs = []  # type:List[TeaCacheConfig]

        import json

        config = {}
        try:
            with open(path, "r") as fp:
                config = dict(json.load(fp))
        except Exception:
            config = {}

        if keys is not None:
            for key in keys:
                config = config.get(key, {})

        for name, data in config.items():
            json_data = None
            if name == "meta_data":
                json_data = data
            else:
                json_data = data.get("meta_data", None)

            teacache_config = TeaCacheConfig()
            if teacache_config.from_json(json_data):
                teacache_configs.append(teacache_config)

        if len(teacache_configs) < 1:
            return None

        if only_valid:
            teacache_configs = [config for config in teacache_configs if config.valid()]

        return teacache_configs

    def save_file(self, path: str, keys: List = None):
        from pathlib import Path
        import os, json

        save_path = Path(path)
        os.makedirs(save_path.parent, exist_ok=True)

        config = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as fp:
                    config = json.load(fp)
            except Exception as ex:
                print(f"fail to analyse teacache config from {path}, skip:", str(ex))
                config = {}

        ptr = config
        if keys is not None and len(keys) > 0:
            for key in keys:
                if key in ptr:
                    ptr = ptr[key]
                else:
                    ptr[key] = {}
                    ptr = ptr[key]
        ptr["meta_data"] = self.to_json()

        try:
            with open(path, "w") as fp:
                json.dump(config, fp, indent=4)
        except Exception as ex:
            print("fail to save teacache config:", str(ex))


class TeaCacheSolver:
    def __init__(
        self, max_points_count=100, min_points_count=5, fit_length=0, enable=True
    ):
        self.min_points_count = min_points_count
        self.max_points_count = max_points_count
        self.fit_length = fit_length

        self.record_points = deque(maxlen=max_points_count)

        self.step_cache = (
            float("-inf"),
            None,
            None,
        )  # type: Tuple[int,torch.Tensor,torch.Tensor]

        self.enable = enable

    @property
    def points_count(self):
        return len(self.record_points)

    def ready(self, must_max=False):
        if must_max:
            return self.points_count >= self.max_points_count
        else:
            return self.points_count >= self.min_points_count

    def add_point(self, point: Tuple[float]):
        self.record_points.append(point)

    def add_points(self, points: Tuple[Tuple[float]]):
        for point in points:
            self.add_point(point)

    def points(self) -> Tuple[np.ndarray]:
        xs = np.array([point[0] for point in self.record_points])
        ys = np.array([point[1] for point in self.record_points])
        return xs, ys

    def create_config(self, complexity=None) -> TeaCacheConfig:
        if not self.enable:
            return None

        config = TeaCacheConfig(
            complexity=(
                complexity if complexity is not None else self.min_points_count - 1
            ),
            fit_length=self.fit_length,
        )

        config, success = self.update_config(config)
        if success:
            return config
        else:
            return None

    def update_config(
        self, config: TeaCacheConfig = None
    ) -> Tuple[TeaCacheConfig, bool]:
        if self.points_count < self.min_points_count:
            return config, False

        if config is None:
            config = TeaCacheConfig(
                complexity=self.min_points_count - 1, fit_length=self.fit_length
            )

        config.update_func(self.points())
        return config, True

    def set_step_cache(
        self, step: int, input: torch.Tensor, output: torch.Tensor
    ) -> bool:
        if not self.enable:
            return False

        previous_step, previous_input, previous_output = self.step_cache
        self.step_cache = (step, input, output)
        if previous_step + 1 != step:
            return False
        else:
            x = (
                ((input - previous_input).abs().mean() / previous_input.abs().mean())
                .cpu()
                .item()
            )

            y = (
                ((output - previous_output).abs().mean() / previous_output.abs().mean())
                .cpu()
                .item()
            )

            self.add_point((x, y))
            return True

    def clear_step_cache(self):
        self.step_cache = (
            float("-inf"),
            None,
            None,
        )  # type: Tuple[int,torch.Tensor,torch.Tensor]


class TeaCacheSolvers:
    def __init__(self, max_points_count=200, min_points_count=5, enable=True):
        self.min_points_count = min_points_count
        self.max_points_count = max_points_count
        self.solvers = {}  # type: Dict[str,Dict[str,TeaCacheSolver]]

        self.enable = enable

    def set_enable(self, enable: bool):
        self.enable = enable

    def set_step_cache(
        self,
        keys: list,
        step: int,
        fit_length: int,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> bool:
        if not self.enable:
            return False

        assert fit_length > 0, "sequence length must >0"

        keys = str(keys)
        if keys not in self.solvers:
            self.solvers[keys] = {}

        if str(fit_length) not in self.solvers[keys]:
            self.solvers[keys][str(fit_length)] = TeaCacheSolver(
                max_points_count=self.max_points_count,
                min_points_count=self.min_points_count,
                fit_length=fit_length,
            )

        return self.solvers[keys][str(fit_length)].set_step_cache(step, input, output)

    def save_configs(
        self, path: str = "config/teacache/cache.json", complexity=4, must_max=False
    ) -> int:
        if not self.enable:
            return False
        count = 0
        for keys in self.solvers:
            for fit_length, solver in self.solvers[keys].items():
                if solver.ready(must_max=must_max):
                    config = solver.create_config(complexity=complexity)
                    if config is not None:
                        config.save_file(
                            path,
                            keys=[*eval(keys), str(fit_length)],
                        )
                        count += 1

        return count

    def clear_step_caches(self):
        if not self.enable:
            return False

        for keys in self.solvers:
            for _, solver in self.solvers[keys].items():
                solver.clear_step_cache()


class TeaCache:
    def __init__(
        self,
        max_skip_step: int,
        min_skip_step: int = 0,
        max_consecutive_skip: int = 2,
        threshold: float = 0.0,
        configs: TeaCacheConfig | List[TeaCacheConfig] = None,
        model_keys: str | List[str] = None,
        cache_path="config/teacache/cache.json",
        speedup_mode=True,
        enable: bool = True,
    ):
        self.cache_path = cache_path
        self.model_keys = model_keys if isinstance(model_keys, list) else [model_keys]
        self.enable = enable
        self.speedup_mode = speedup_mode

        self.sum_l1_distance = 0
        self.rel_l1_threshold = threshold
        self.max_skip_step = max_skip_step
        self.min_skip_step = min_skip_step
        self.max_consecutive_skip = max_consecutive_skip

        if isinstance(configs, TeaCacheConfig):
            configs = [configs]

        if configs is None:
            self.configs = TeaCacheConfig.load_files(
                path=cache_path, keys=self.model_keys
            )
        else:
            self.configs = configs

        if self.enable and self.rel_l1_threshold > 1e-5 and self.speedup_mode:
            if self.configs is None or len(self.configs) < 1:
                raise ValueError(
                    "teacache need config args, you can also refer by model_name with cache_path"
                )

        self.pre_step = -float("inf")
        self.previous_calc_step = -float("inf")
        self.previous_residual = None  # type:torch.Tensor
        self.previous_t_mod = None  # type:torch.Tensor
        self.cache_l1_distance = (0, 0.0)

        if not self.speedup_mode:
            self.solver = TeaCacheSolvers()
        else:
            self.solver = None

        self.skip_steps_sum = 0
        self.calc_steps_sum = 0

    def current_speedup_rate(self) -> float:
        if self.skip_steps_sum + self.calc_steps_sum < 1:
            return 1
        elif self.calc_steps_sum < 1:
            return float("inf")
        else:
            return (self.skip_steps_sum + self.calc_steps_sum) / self.calc_steps_sum

    def config(self, sequence_length) -> TeaCacheConfig:
        return min(self.configs, key=lambda x: x.matching_rate(sequence_length))

    def do_speed(self):
        return (
            self.rel_l1_threshold > 1e-5
            and self.configs is not None
            and len(self.configs) > 0
            and self.speedup_mode
            and self.enable
        )

    def set_range(
        self,
        max_skip_step: int,
        min_skip_step: int = 0,
        threshold: float = 0,
        max_consecutive_skip: int = None,
    ):
        self.max_skip_step = max_skip_step
        self.min_skip_step = min_skip_step
        self.rel_l1_threshold = threshold

        if max_consecutive_skip is not None:
            self.max_consecutive_skip = max_consecutive_skip

    def do_solver(self):
        return not self.speedup_mode and self.enable and self.solver is not None

    def check(
        self,
        step: int,
        t_mod: torch.Tensor,
        sequence_length: int = 0,
    ) -> bool:
        self.cache_l1_distance = (0, 0.0)
        if not self.do_speed():
            return False

        # check step valid
        if (
            step < self.min_skip_step
            or step > self.max_skip_step
            or self.pre_step + 1 != step
            or (step - self.previous_calc_step) > self.max_consecutive_skip
        ):
            return False

        self.cache_l1_distance = (
            step,
            self.config(sequence_length).evaluate(
                (
                    (t_mod - self.previous_t_mod).abs().mean()
                    / self.previous_t_mod.abs().mean()
                )
                .cpu()
                .item()
            ),
        )

        if self.cache_l1_distance[1] + self.sum_l1_distance < self.rel_l1_threshold:
            return True
        else:
            return False

    def store_truth(
        self,
        step: int,
        t_mod: torch.Tensor,
        input_latent: torch.Tensor,
        output_latent: torch.Tensor,
        sequence_length: int = None,
    ):
        if self.do_solver():
            self.solver.set_step_cache(
                keys=self.model_keys,
                step=step,
                fit_length=sequence_length,
                input=input_latent,
                output=output_latent,
            )

            self.solver.save_configs(path=self.cache_path, complexity=4, must_max=False)

        if not self.do_speed():
            return None

        if step == 0:
            self.skip_steps_sum = 0
            self.calc_steps_sum = 1
        else:
            self.calc_steps_sum += 1

        self.previous_residual = output_latent - input_latent
        self.sum_l1_distance = 0.0
        self.previous_t_mod = t_mod
        self.pre_step = step
        self.previous_calc_step = step
        self.cache_l1_distance = (0, 0.0)

    def update(self, t_mod: torch.Tensor, input_latent: torch.Tensor):
        if not self.do_speed():
            return input_latent

        assert (
            self.cache_l1_distance[0] == self.pre_step + 1
        ), "internal error, cache bad data"

        self.sum_l1_distance += self.cache_l1_distance[1]
        self.pre_step += 1
        self.cache_l1_distance = (0, 0.0)
        self.previous_t_mod = t_mod
        output_latent = input_latent + self.previous_residual

        self.skip_steps_sum += 1

        return output_latent
