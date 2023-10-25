"""
custom utilities for yolo9000-like (shallow) hierarchical class training

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

import torch


def get_class_tree(names: list[str], tree: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    tree_idx = torch.tensor(
        [names.index(x) if x != '' else -1 for x in tree], dtype=torch.long)
    class_tree = {'root': torch.nonzero(tree_idx == -1).flatten()}
    tree_map = torch.empty_like(tree_idx)
    tree_map.copy_(tree_idx)
    tree_map[tree_idx == -1] = class_tree['root']
    class_tree['tree'] = tree_map

    return dict_to_device(class_tree, device)


def dict_to_device(data: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """ move all torch.Tensors in dictionary to device """
    return {key: data[key].to(device) if isinstance(data[key], torch.Tensor) else data[key] for key in data.keys()}
