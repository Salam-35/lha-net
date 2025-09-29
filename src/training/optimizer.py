import torch
import torch.optim as optim
from typing import Dict, Any, Optional, List
import math


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    amsgrad: bool = False,
    differential_lr: bool = True,
    backbone_lr_factor: float = 0.1,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with optional differential learning rates

    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd', 'rmsprop')
        learning_rate: Base learning rate
        weight_decay: Weight decay factor
        momentum: Momentum for SGD
        betas: Betas for Adam/AdamW
        eps: Epsilon for numerical stability
        amsgrad: Use AMSGrad variant for Adam
        differential_lr: Use different learning rates for different parts
        backbone_lr_factor: Learning rate factor for backbone (usually lower)

    Returns:
        Configured optimizer
    """

    if differential_lr:
        # Group parameters by component
        param_groups = _create_parameter_groups(
            model, learning_rate, weight_decay, backbone_lr_factor
        )
    else:
        # Single parameter group
        param_groups = [{'params': model.parameters(), 'lr': learning_rate, 'weight_decay': weight_decay}]

    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )

    elif optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )

    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )

    elif optimizer_type.lower() == "rmsprop":
        optimizer = optim.RMSprop(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps
        )

    elif optimizer_type.lower() == "adagrad":
        optimizer = optim.Adagrad(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=eps
        )

    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer


def _create_parameter_groups(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    backbone_lr_factor: float
) -> List[Dict[str, Any]]:
    """Create parameter groups with differential learning rates"""

    backbone_params = []
    pmsa_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'backbone' in name:
            backbone_params.append(param)
        elif 'pmsa' in name or 'hierarchical_pmsa' in name:
            pmsa_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = []

    # Backbone parameters (lower learning rate)
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': base_lr * backbone_lr_factor,
            'weight_decay': weight_decay,
            'name': 'backbone'
        })

    # PMSA parameters (base learning rate)
    if pmsa_params:
        param_groups.append({
            'params': pmsa_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'pmsa'
        })

    # Decoder parameters (slightly higher learning rate)
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': base_lr * 1.5,
            'weight_decay': weight_decay * 0.5,  # Lower weight decay for decoder
            'name': 'decoder'
        })

    # Other parameters (base learning rate)
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'other'
        })

    return param_groups


class LookAhead:
    """Look Ahead optimizer wrapper"""

    def __init__(self, optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0

        # Store slow weights
        self.slow_weights = {}
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    self.slow_weights[param] = param.data.clone()

    def step(self):
        """Perform optimizer step with look ahead"""
        self.optimizer.step()
        self.step_count += 1

        if self.step_count % self.k == 0:
            # Update slow weights
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        slow_weight = self.slow_weights[param]
                        # slow_weight = slow_weight + alpha * (fast_weight - slow_weight)
                        slow_weight.add_(param.data - slow_weight, alpha=self.alpha)
                        param.data.copy_(slow_weight)

    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Get state dict"""
        return {
            'optimizer': self.optimizer.state_dict(),
            'slow_weights': self.slow_weights,
            'step_count': self.step_count
        }

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.slow_weights = state_dict['slow_weights']
        self.step_count = state_dict['step_count']


class EMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device

        # Create EMA model
        self.ema_model = type(model)(
            **{k: v for k, v in model.__dict__.items() if not k.startswith('_')}
        ).to(self.device)

        # Initialize EMA parameters
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_model.eval()

        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self):
        """Update EMA parameters"""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.mul_(self.decay).add_(model_param, alpha=1 - self.decay)

    def apply_ema(self):
        """Apply EMA weights to the original model"""
        with torch.no_grad():
            for model_param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                model_param.copy_(ema_param)

    def restore(self):
        """Restore original model weights"""
        # This would require storing original weights, which we don't do here
        # for memory efficiency. Instead, the user should save/load checkpoints.
        pass

    def state_dict(self):
        """Get state dict"""
        return {
            'ema_model': self.ema_model.state_dict(),
            'decay': self.decay
        }

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.decay = state_dict['decay']


class OptimizerFactory:
    """Factory for creating optimizers with common configurations"""

    @staticmethod
    def create_lha_net_optimizer(
        model: torch.nn.Module,
        config: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Create optimizer specifically configured for LHA-Net"""

        default_config = {
            'optimizer_type': 'adamw',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'differential_lr': True,
            'backbone_lr_factor': 0.1,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }

        # Update with provided config
        default_config.update(config)

        return create_optimizer(model, **default_config)

    @staticmethod
    def create_optimizer_with_lookahead(
        model: torch.nn.Module,
        config: Dict[str, Any],
        lookahead_k: int = 5,
        lookahead_alpha: float = 0.5
    ) -> LookAhead:
        """Create optimizer with LookAhead wrapper"""

        base_optimizer = OptimizerFactory.create_lha_net_optimizer(model, config)
        return LookAhead(base_optimizer, k=lookahead_k, alpha=lookahead_alpha)

    @staticmethod
    def create_optimizer_with_ema(
        model: torch.nn.Module,
        config: Dict[str, Any],
        ema_decay: float = 0.999
    ) -> tuple[torch.optim.Optimizer, EMA]:
        """Create optimizer with EMA"""

        optimizer = OptimizerFactory.create_lha_net_optimizer(model, config)
        ema = EMA(model, decay=ema_decay)

        return optimizer, ema