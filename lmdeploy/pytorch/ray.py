# Copyright (c) OpenMMLab. All rights reserved.
import os
import time

import ray
from ray.util.placement_group import PlacementGroup

from lmdeploy.pytorch.devices import get_device_manager
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
PG_WAIT_TIMEOUT = 1800


def get_device_str(device_type: str = None) -> str:
    """Get device str."""
    device_type = device_type or get_device_manager().current_context().device_type
    if device_type in ['cuda', 'maca']:
        device_type = 'GPU'
    elif device_type == 'ascend':
        device_type = 'NPU'
    elif device_type == 'camb':
        device_type = 'MLU'
    else:
        raise ValueError(f'Unsupported device type: {device_type}')

    return device_type


def get_resource_kwargs(device_str: str, resource_used: float = 0.01) -> dict[str, float]:
    """Get resource kwargs."""
    if device_str == 'GPU':
        resource_kwargs = {'num_gpus': resource_used}
    elif device_str == 'NPU':
        resource_kwargs = {'resources': {device_str: resource_used}}
    else:
        raise ValueError(f'Unsupported device type: {device_str}')
    return resource_kwargs


def _infer_local_ray_custom_resources(device_type: str, world_size: int) -> dict[str, float] | None:
    """Resources to pass to ray.init() for a fresh local cluster.

    Ray does not auto-detect Ascend NPUs (or Camb MLUs); without registering
    custom resources, placement groups requesting e.g. ``{'NPU': 1}`` never schedule.
    """
    if device_type == 'ascend':
        n = None
        try:
            import torch
            npu_mod = getattr(torch, 'npu', None)
            if npu_mod is not None and callable(getattr(npu_mod, 'device_count', None)):
                n = int(npu_mod.device_count())
                if n <= 0:
                    n = None
        except Exception:
            n = None
        if n is None:
            vis = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '').strip()
            if vis:
                n = len([x for x in vis.split(',') if x.strip() != ''])
        if n is None or n <= 0:
            n = int(world_size)
            logger.warning(
                'Could not detect NPU count via torch.npu.device_count() or '
                'ASCEND_RT_VISIBLE_DEVICES; registering Ray resource NPU=%d from '
                'parallel world size. Set visible devices if this is wrong.', n)
        elif n < world_size:
            logger.warning(
                'Detected %d NPUs on node but placement group needs %d bundles; '
                'scheduling may fail. Check tp/world_size and device visibility.', n,
                world_size)
        return {'NPU': float(n)}
    if device_type == 'camb':
        n = None
        try:
            import torch
            mlu = getattr(torch, 'mlu', None)
            if mlu is not None and callable(getattr(mlu, 'device_count', None)):
                n = int(mlu.device_count())
                if n <= 0:
                    n = None
        except Exception:
            n = None
        if n is None or n <= 0:
            n = int(world_size)
            logger.warning(
                'Could not detect MLU count; registering Ray resource MLU=%d from world_size.', n)
        return {'MLU': float(n)}
    return None


def _wait_until_pg_ready(current_placement_group: PlacementGroup):
    """Wait until a placement group is ready.

    It prints the informative log messages if the placement group is not created within time.
    """
    # copy from vLLM
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    placement_group_specs = current_placement_group.bundle_specs

    s = time.time()
    pg_ready_ref = current_placement_group.ready()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        if len(ready) > 0:
            break

        # Exponential backoff for warning print.
        wait_interval *= 2
        logger.info(
            'Waiting for creating a placement group of specs for '
            '%d seconds. specs=%s. Check '
            '`ray status` to see if you have enough resources,'
            ' and make sure the IP addresses used by ray cluster'
            ' are the same as VLLM_HOST_IP environment variable'
            ' specified in each node if you are running on a multi-node.', int(time.time() - s), placement_group_specs)

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        raise ValueError('Cannot provide a placement group of '
                         f'{placement_group_specs=} within {PG_WAIT_TIMEOUT} seconds. See '
                         '`ray status` to make sure the cluster has enough resources.') from None


def _get_obj_store_memory(dp: int = 1):
    """Get obj store memory."""
    import psutil
    DEFAULT_OBJECT_STORE_MEMORY_PROPORTION = os.getenv('RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION', '0.3')
    DEFAULT_OBJECT_STORE_MEMORY_PROPORTION = float(DEFAULT_OBJECT_STORE_MEMORY_PROPORTION)
    DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES = os.getenv('RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES', None)
    if DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES is None:
        DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES = 80 * (10**9)
    else:
        DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES = int(DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES)
    total_mem = psutil.virtual_memory().total
    obj_store_mem = int(total_mem * DEFAULT_OBJECT_STORE_MEMORY_PROPORTION)
    obj_store_mem = min(DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES, obj_store_mem)
    if dp > 1:
        obj_store_mem = obj_store_mem // min(8, dp)
    return obj_store_mem


def init_ray_cluster(world_size: int, ray_address: str = None, dp: int = 1, device_type: str = 'cuda'):
    """Init ray cluster."""
    # modifier from vLLM
    if not ray.is_initialized():
        num_cpus = world_size
        object_store_memory = _get_obj_store_memory(dp=dp)
        init_kwargs = dict(
            ignore_reinit_error=True,
            num_cpus=num_cpus,
            object_store_memory=object_store_memory,
        )
        if ray_address is not None:
            init_kwargs['address'] = ray_address
        if ray_address is None:
            custom_res = _infer_local_ray_custom_resources(device_type, world_size)
            if custom_res:
                init_kwargs['resources'] = custom_res
        try:
            ray.init(**init_kwargs)
        except ValueError as e:
            if e.args is not None and len(e.args) >= 1 and e.args[
                    0] == 'When connecting to an existing cluster, num_cpus and num_gpus must not be provided.':
                ray.init(address=ray_address, ignore_reinit_error=True)
            else:
                raise

    device_str = get_device_str(device_type)

    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    owned_pg = False
    if not current_placement_group:
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        if world_size > num_devices_in_cluster:
            logger.warning(
                'The number of required %ss exceeds the total '
                'number of available %ss in the placement group.', device_str, device_str)
        # Create a new placement group
        placement_group_specs: list[dict[str, float]] = ([{device_str: 1.0} for _ in range(world_size)])

        # Pin at least one bundle to the local node.
        # This helps multi-node DP keep each dp_rank process's workers co-located with
        # the node where the process is launched.
        current_ip = ray.util.get_node_ip_address()
        placement_group_specs[0][f'node:{current_ip}'] = 0.001

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(placement_group_specs, strategy='PACK')
        _wait_until_pg_ready(current_placement_group)
        owned_pg = True

    assert current_placement_group is not None
    # Set the placement group in the parallel config
    placement_group = current_placement_group
    return placement_group, owned_pg


class RayContext:
    """Context manager for Ray."""

    def __init__(self, world_size: int, ray_address: str = None, dp: int = 1, device_type: str = 'cuda'):
        """Initialize Ray context."""
        placement_group, owned_pg = init_ray_cluster(world_size=world_size,
                                                     ray_address=ray_address,
                                                     dp=dp,
                                                     device_type=device_type)

        self.placement_group = placement_group
        self.owned_pg = owned_pg

    def get_placement_group(self):
        """Get the placement group."""
        return self.placement_group

    def shutdown(self):
        """Shutdown Ray."""
        if self.owned_pg:
            ray.util.remove_placement_group(self.placement_group)
            logger.debug('RayContext placement group removed.')

        if ray.is_initialized():
            try:
                ray.shutdown()
                logger.debug('Ray shutdown.')
            except Exception:
                logger.exception('Error during Ray shutdown.')
        else:
            logger.debug('Ray is not initialized, skipping shutdown.')
