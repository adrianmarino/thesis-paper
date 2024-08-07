import torch
import pytorch_common.util as pu

def indexes_of(tensor, value): 
    return ((tensor == value).nonzero(as_tuple=True)[0])


def random_int(begin, end, size):
    return torch.randint(begin, end, (size,))

def random_choice(tensor, size):
    n = tensor.size()[0]-1 if tensor.size()[0] > 1 else 1
    return tensor[random_int(0, n, size)]


def apply(tensor, fn):
    copy = torch.clone(tensor)
    copy.apply_(fn)
    return copy


def is_int(tensor):
    return tensor.dtype == torch.uint8 \
        or tensor.dtype == torch.int8 \
        or tensor.dtype == torch.int16 \
        or tensor.dtype == torch.int32 \
        or tensor.dtype == torch.int64


def delete(tensor):
    tensor.detach()
    del tensor
    free_gpu_memory()


def free_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        

def value_counts(tensor):
    values, counts = tensor.unique(return_counts=True)
    return {values[idx].item(): counts[idx].item() for idx in range(len(values))}