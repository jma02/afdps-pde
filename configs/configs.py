from dataclasses import dataclass, asdict, field


@dataclass
class TrainConfig:
    seed: int = 159753
    batch_size: int = 128
    num_workers: int = 2
    lr: float = 1e-4
    optimizer: str = 'adam'
    precision: str = '32-true'
    warmup_steps: int = 10
    clip_grad: bool = True
    grad_clip_val: float = 1.0
    epochs: int = 150
    log_interval: int = 50

@dataclass
class UnetConfig:
    ch: int = 128
    ch_mul: list[int] = field(default_factory=lambda: [1, 2, 2, 2])
    att_channels: list[int] = field(default_factory=lambda: [0, 1, 0, 0])
    dropout: float = 0.1
    
@dataclass
class RunConfig:
    dataset: str = 'cifar10'
    scale_shift: bool = False
    scheduler_type: str = 'linear'
    timesteps: int = 1000
    guidance: bool = False
    num_classes: int = -1
    p_cond: float = -1


def make_config(file = None):
    if file is None:
        config = {
            'train': asdict(TrainConfig()),
            'unet': asdict(UnetConfig()),
            'run': asdict(RunConfig())
        }

        return config

    import yaml

    file = 'configs/' + file + '.yml'

    with open(file, 'r') as f:
        yml_dict = yaml.load(f, Loader = yaml.SafeLoader)
    
    config = {}

    config['train'] = asdict(TrainConfig(**yml_dict.get('train', {})))
    config['unet'] = asdict(UnetConfig(**yml_dict.get('unet', {})))
    config['run'] = asdict(RunConfig(**yml_dict.get('run', {})))

    return config