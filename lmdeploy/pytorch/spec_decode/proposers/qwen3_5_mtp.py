# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

from .base import SPEC_PROPOSERS
from .deepseek_mtp import DeepseekMTP

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='qwen3_5_mtp')
class Qwen3_5MTP(DeepseekMTP):

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        logger.info('Using embed_tokens from target model.')
        # vLLM MTP invariant: MTP shares embed_tokens and lm_head with target.
        # We already share embed_tokens here; lm_head sharing is handled by
        # the base class build_model implementation.
        target_emb = target_model.get_input_embeddings()
        model = self.model
        if hasattr(model, 'set_input_embeddings'):
            model.set_input_embeddings(target_emb)
        elif hasattr(model, 'model') and hasattr(model.model, 'set_input_embeddings'):
            model.model.set_input_embeddings(target_emb)
        elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            # lmdeploy Qwen3_5 / MoE stacks are nn.Module, not PreTrainedModel
            model.model.language_model.embed_tokens = target_emb
        else:
            raise AttributeError(
                'Draft model has no set_input_embeddings and no language_model.embed_tokens; '
                f'got {type(model).__name__}.')
        assert model.get_input_embeddings() is not None, 'Input embeddings should not be None.'
