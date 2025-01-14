from collections import OrderedDict

import torch.distributed as dist
from det3d.torchie.trainer import OptimizerHook
from torch._utils import _flatten_dense_tensors, _take_tensors, _unflatten_dense_tensors


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
            bucket, _unflatten_dense_tensors(flat_tensors, bucket)
        ):
            tensor.copy_(synced)


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    grads = [
        param.grad.data
        for param in params
        if param.requires_grad and param.grad is not None
    ]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


# import torch
class DistOptimizerHook(OptimizerHook):
    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        ### losss NAN debug
        # import torch.autograd as autograd
        # print("$"*100)
            # with autograd.detect_anomaly():
        # print("$" * 100)
        # print(runner.outputs)
        runner.outputs["loss"].backward()
        # for name, param in runner.model.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print("nan gradient found")
        #         print("name:", name)
                # print("param:", param.grad)
                # raise SystemExit
        allreduce_grads(runner.model.parameters(), self.coalesce, self.bucket_size_mb)
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()
