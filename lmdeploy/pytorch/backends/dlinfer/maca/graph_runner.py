from lmdeploy.pytorch.backends.cuda.graph_runner import CUDAGraphRunner

from typing import List

import torch
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig


class MacaGraphRunner(CUDAGraphRunner):

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                 backend_config: BackendConfig, device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config, device)
        self.enable_graph_mode = self.backend_config.eager_mode == False