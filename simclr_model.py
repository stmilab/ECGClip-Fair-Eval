import torch.nn as nn
from resnet1d import ResNet1D
import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
import re


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LinearLayer(nn.Module):
    def __init__(
        self, in_features, out_features, use_bias=True, use_bn=False, **kwargs
    ):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.linear = nn.Linear(
            self.in_features, self.out_features, bias=self.use_bias and not self.use_bn
        )
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        head_type="nonlinear",
        **kwargs
    ):
        super(ProjectionHead, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == "linear":
            self.layers = LinearLayer(self.in_features, self.out_features, False, True)
        elif self.head_type == "nonlinear":
            self.layers = nn.Sequential(
                LinearLayer(self.in_features, self.hidden_features, True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features, self.out_features, False, True),
            )

    def forward(self, x):
        x = self.layers(x)
        return x


class PreModel(nn.Module):
    # no base model for now

    # def __init__(self,base_model):
    def __init__(self):
        super().__init__()
        # self.base_model = base_model

        # parameters - need to change

        kernel_size = 16
        stride = 2
        n_block = 48
        downsample_gap = 6
        increasefilter_gap = 12

        # PRETRAINED MODEL
        self.pretrained = ResNet1D(
            in_channels=1,
            base_filters=64,  # 64 for ResNet1D, 352 for ResNeXt1D
            kernel_size=kernel_size,
            stride=stride,
            groups=32,
            n_block=n_block,
            n_classes=1024,
            downsample_gap=downsample_gap,
            increasefilter_gap=increasefilter_gap,
            use_do=True,
        )

        # self.pretrained = models.resnet50(pretrained=True)
        # self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        # self.pretrained.maxpool = Identity()

        self.pretrained.fc = Identity()

        for p in self.pretrained.parameters():
            p.requires_grad = True

        self.projector = ProjectionHead(1024, 1024, 64)

    def forward(self, x):
        out = self.pretrained(x)
        xp = self.projector(torch.squeeze(out[0]))

        return xp


class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        # SIMCLR
        labels = (
            torch.from_numpy(np.array([0] * N))
            .reshape(-1)
            .to(positive_samples.device)
            .long()
        )  # .float()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


EETA_DEFAULT = 0.001


class SupConLoss(
    nn.Module
):  # Code Frrom https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        print("Features", features)
        print("labels", labels.shape)
        print("mask", mask)
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        print("features before", features)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        print("contrast_feature1", contrast_feature)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count

            print("anchor_feature", anchor_feature)
            print("anchor_count", anchor_count)
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        print("anchor_dot_contrast", anchor_dot_contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        print("logits", logits)
        print("mask.shape1", mask.shape)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )

        mask = mask * logits_mask
        print("MASKKKKKKK", mask)
        print("Max Logits", torch.max(logits))
        print("Min Logits", torch.min(logits))
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        print("Max exp_logits", torch.max(exp_logits))
        print("Min exp_logits", torch.min(exp_logits))
        print("exp_logits", exp_logits)
        print("log_prob", log_prob)
        print("isnan?!", torch.sum(torch.isnan(log_prob).int()))
        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        print("isnan?!2", torch.sum(torch.isnan(mask).int()))
        print("mask_pos_pairs", mask_pos_pairs)
        print("(mask * log_prob).sum(1) ", (mask * log_prob).sum(1))
        print("max", torch.max(log_prob))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(
        #     1
        # )  # the new one from here: https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/losses.py
        print("mean_log_prob_pos", mean_log_prob_pos)
        print("isnan?!3", torch.sum(torch.isnan(mean_log_prob_pos).int()))
        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        print(
            "(self.temperature / self.base_temperature)",
            (self.temperature / self.base_temperature),
        )
        print(
            "-(self.temperature / self.base_temperature) * mean_log_prob_pos",
            -(self.temperature / self.base_temperature) * mean_log_prob_pos,
        )
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.gt(0),
                        torch.where(
                            g_norm.gt(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True


class FCNet(nn.Module):
    def __init__(self, in_shape):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(in_shape, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        return self.fc2(self.fc1(x))
