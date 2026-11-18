"""One-batch lookahead for pinned training tensors on a dedicated copy stream."""
import torch


def move_batch(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch(v, device) for k, v in batch.items()}
    return batch


def _record_batch(batch, stream):
    if isinstance(batch, dict):
        for value in batch.values():
            _record_batch(value, stream)
    elif torch.is_tensor(batch):
        batch.record_stream(stream)


def device_batches(loader, device, *, prefetch=True):
    if not prefetch or device.type != 'cuda':
        for batch in loader:
            yield move_batch(batch, device)
        return

    copy_stream = torch.cuda.Stream(device=device)
    def copy(batch):
        with torch.cuda.stream(copy_stream):
            value = move_batch(batch, device)
            return value, copy_stream.record_event()

    iterator = iter(loader)
    host_batch = next(iterator, None)
    if host_batch is None:
        return
    pending = copy(host_batch)
    while pending is not None:
        batch, ready = pending
        host_batch = next(iterator, None)
        pending = None if host_batch is None else copy(host_batch)
        consumer = torch.cuda.current_stream(device)
        consumer.wait_event(ready)
        _record_batch(batch, consumer)
        yield batch
