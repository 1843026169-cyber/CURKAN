import math
import torch
from torch import nn
import torch.nn.functional as F

class TaylorKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        order=3,
        scale_base=1.0,
        scale_taylor=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    ):

        super(TaylorKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.scale_base = scale_base
        self.scale_taylor = scale_taylor
        self.base_activation = base_activation()
        self.use_bias = use_bias
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.taylor_coeffs = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, order)
        )

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            std = self.scale_taylor / (self.in_features * math.sqrt(self.order))
            self.taylor_coeffs.normal_(mean=0.0, std=std)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def taylor_series(self, x: torch.Tensor):

        x_expanded = x.unsqueeze(1).unsqueeze(-1)
        powers = torch.arange(self.order, device=x.device).view(1, 1, 1, -1)
        x_powers = x_expanded ** powers

        taylor_coeffs_expanded = self.taylor_coeffs.unsqueeze(0)
        taylor_terms = x_powers * taylor_coeffs_expanded
        taylor_output = taylor_terms.sum(dim=3).sum(dim=2)

        return taylor_output

    def forward(self, x: torch.Tensor):

        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        taylor_output = self.taylor_series(x)
        output = base_output + taylor_output

        if self.use_bias:
            output += self.bias
        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):

        coeffs_l2 = self.taylor_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2


def recursive_getattr(model, module_name):

    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):

    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)


class LinearLayer_curkan(nn.Module):

    def __init__(self,
                 weight,
                 lora_dim = 0,
                 lora_scaling = 1,
                 lora_dropout = 0,
                 bias = None):
        super(LinearLayer_curkan, self).__init__()
        self.weight = weight
        self.bias = bias
        self.lora_dim = lora_dim

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use curLoRA, whose reduced dim should be larger than 1"
            )

        try:
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape

        self.lora_C_weight = nn.Parameter(torch.zeros(rows, self.lora_dim))
        self.lora_R_weight = nn.Parameter(torch.zeros(self.lora_dim, columns))
        self.lora_scaling = lora_scaling / self.lora_dim

        self.in_dim = lora_dim
        self.out_dim = lora_dim

        hidden_dim = lora_dim

        self.kan1 = TaylorKANLinear(
            in_features = self.in_dim,
            out_features = hidden_dim,

        )

        self.kan2 = TaylorKANLinear(
            in_features = hidden_dim,
            out_features = self.out_dim,
        )

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        self.weight.requires_grad = False
        self.lora_C_weight.requires_grad = False
        self.lora_R_weight.requires_grad = False
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode = True):
        self.lora_dropout.train(mode)

    def reset_parameters(self):

        col_norms = torch.norm(self.weight, p = 2, dim = 0)
        col_indices = torch.topk(col_norms, self.lora_dim, largest = True).indices
        self.lora_C_weight.data = self.weight[:, col_indices]

        row_norms = torch.norm(self.lora_C_weight, p = 2, dim = 1)
        row_indices = torch.topk(row_norms, self.lora_dim, largest = True).indices
        self.lora_R_weight.data = self.weight[row_indices, :]

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(torch.matmul(self.lora_C_weight, self.lora_R_weight))
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(torch.matmul(self.lora_C_weight, self.lora_R_weight))
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:

            x1 = self.lora_dropout(input) @ self.lora_R_weight.t()
            x2 = x1.reshape(-1, self.lora_dim)
            x3 = self.kan1(x2)
            x4 = self.kan2(x3)
            x5 = x4.reshape(x1.shape[0], -1, self.lora_dim)
            x = x5 @ self.lora_C_weight.t()

            return (F.linear(input, self.weight, self.bias) +
                    x * self.lora_scaling)

def convert_linear_layer_to_curkan(model,
                                   part_module_name,
                                   lora_dim = 0,
                                   lora_scaling = 1,
                                   lora_dropout = 0):
    replace_name = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if isinstance(part_module_name, str):
                match = part_module_name in name
            else:
                match = any(key in name for key in part_module_name)

            if match:
                replace_name.append(name)

    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_curkan(
            module.weight, lora_dim, lora_scaling, lora_dropout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model

def convert_curlora_to_linear_layer(model):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_curkan):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        module.fuse_lora_weight()
    return model


def only_optimize_lora_parameters(model, force_optimize_params = []):

    for name, param in model.named_parameters():
        if "kan1" in name or "kan2" in name or name in force_optimize_params:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def make_model_gradient_checkpointing_compatible(model):

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model

def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr = 5e-4,
        no_decay_name_list = [
            "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
            "ln_f.weight"
        ],
        lora_name_list = ["kan1", "kan2"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower()
                                                for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
            "lr":
                lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups