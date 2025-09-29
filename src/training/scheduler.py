import torch
import torch.optim as optim
from typing import Dict, Any, Optional, Union
import math
import warnings


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_warmup",
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        num_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate

    Returns:
        Configured scheduler
    """

    if scheduler_type.lower() == "cosine_warmup":
        return CosineAnnealingWarmupScheduler(
            optimizer=optimizer,
            total_epochs=num_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
            **kwargs
        )

    elif scheduler_type.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_epochs,
            eta_min=min_lr
        )

    elif scheduler_type.lower() == "step":
        step_size = kwargs.get('step_size', num_epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma
        )

    elif scheduler_type.lower() == "multistep":
        milestones = kwargs.get('milestones', [num_epochs // 3, 2 * num_epochs // 3])
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma
        )

    elif scheduler_type.lower() == "exponential":
        gamma = kwargs.get('gamma', 0.95)
        return optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=gamma
        )

    elif scheduler_type.lower() == "plateau":
        patience = kwargs.get('patience', 10)
        factor = kwargs.get('factor', 0.5)
        threshold = kwargs.get('threshold', 1e-4)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr
        )

    elif scheduler_type.lower() == "polynomial":
        return PolynomialLR(
            optimizer=optimizer,
            total_epochs=num_epochs,
            power=kwargs.get('power', 0.9),
            min_lr=min_lr
        )

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class CosineAnnealingWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with warmup"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        warmup_start_lr: Optional[float] = None,
        last_epoch: int = -1
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

        # Set warmup start LR
        if warmup_start_lr is None:
            self.warmup_start_lr = min_lr
        else:
            self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            cos_epoch = self.last_epoch - self.warmup_epochs
            cos_total = self.total_epochs - self.warmup_epochs

            return [
                self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * cos_epoch / cos_total)) / 2
                for base_lr in self.base_lrs
            ]


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate scheduler"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        power: float = 0.9,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / self.total_epochs) ** self.power
        return [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]


class OneCycleLR(torch.optim.lr_scheduler._LRScheduler):
    """One cycle learning rate scheduler"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: Union[float, list],
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        self.max_lrs = [max_lr] * len(optimizer.param_groups) if isinstance(max_lr, (int, float)) else max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # Calculate initial and final learning rates
        self.initial_lrs = [max_lr / div_factor for max_lr in self.max_lrs]
        self.final_lrs = [initial_lr / final_div_factor for initial_lr in self.initial_lrs]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch

        if step_num <= self.pct_start * self.total_steps:
            # Increasing phase
            phase_step = step_num
            phase_steps = self.pct_start * self.total_steps
            return [
                self.initial_lrs[i] + (self.max_lrs[i] - self.initial_lrs[i]) * phase_step / phase_steps
                for i in range(len(self.max_lrs))
            ]
        else:
            # Decreasing phase
            phase_step = step_num - self.pct_start * self.total_steps
            phase_steps = self.total_steps - self.pct_start * self.total_steps

            if self.anneal_strategy == 'cos':
                cos_factor = (1 + math.cos(math.pi * phase_step / phase_steps)) / 2
                return [
                    self.final_lrs[i] + (self.max_lrs[i] - self.final_lrs[i]) * cos_factor
                    for i in range(len(self.max_lrs))
                ]
            else:  # linear
                return [
                    self.max_lrs[i] + (self.final_lrs[i] - self.max_lrs[i]) * phase_step / phase_steps
                    for i in range(len(self.max_lrs))
                ]


class CyclicLR(torch.optim.lr_scheduler._LRScheduler):
    """Cyclic learning rate scheduler"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: Union[float, list],
        max_lr: Union[float, list],
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[callable] = None,
        scale_mode: str = 'cycle',
        cycle_momentum: bool = True,
        base_momentum: float = 0.8,
        max_momentum: float = 0.9,
        last_epoch: int = -1
    ):
        self.base_lrs = [base_lr] * len(optimizer.param_groups) if isinstance(base_lr, (int, float)) else base_lr
        self.max_lrs = [max_lr] * len(optimizer.param_groups) if isinstance(max_lr, (int, float)) else max_lr

        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.total_size = self.step_size_up + self.step_size_down
        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            self.base_momentums = [base_momentum] * len(optimizer.param_groups)
            self.max_momentums = [max_momentum] * len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle
        if x <= self.step_size_up / self.total_size:
            scale_factor = x * self.total_size / self.step_size_up
        else:
            scale_factor = (x - 1) * self.total_size / self.step_size_down + 1

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (base_momentum - max_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum + base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum + base_height * self.scale_fn(self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs


class SchedulerFactory:
    """Factory for creating schedulers with common configurations"""

    @staticmethod
    def create_lha_net_scheduler(
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any]
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create scheduler specifically configured for LHA-Net training"""

        default_config = {
            'scheduler_type': 'cosine_warmup',
            'num_epochs': 100,
            'warmup_epochs': 5,
            'min_lr': 1e-6
        }

        # Update with provided config
        default_config.update(config)

        return create_scheduler(optimizer, **default_config)

    @staticmethod
    def create_cyclic_scheduler(
        optimizer: torch.optim.Optimizer,
        base_lr: float = 1e-4,
        max_lr: float = 1e-2,
        step_size: int = 2000
    ) -> CyclicLR:
        """Create cyclic learning rate scheduler"""

        return CyclicLR(
            optimizer=optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size,
            mode='triangular2'
        )

    @staticmethod
    def create_one_cycle_scheduler(
        optimizer: torch.optim.Optimizer,
        max_lr: float = 1e-2,
        total_steps: int = 10000,
        pct_start: float = 0.3
    ) -> OneCycleLR:
        """Create one cycle learning rate scheduler"""

        return OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy='cos'
        )