import torch


def load_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a state_dict or a dictionary containing state_dict/model.')
    if checkpoint.get('config', {}).get('research') is not None:
        raise ValueError('RUN2 accepts full-model checkpoints only, not research/ablation checkpoints.')
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    if state and not any(k.startswith('dehazing_model.') for k in state):
        state = {'dehazing_model.' + k: v for k, v in state.items()}
    # Never silently evaluate a partly/randomly initialized model.
    model.load_state_dict(state, strict=True)


