import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear


def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def dice_coefficient(x, target):  # mask_scores的size是[160, 1, 200, 304] x: [160, 200 * 304] target: [160, 200 * 304]
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)  # 160
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union) # [160]
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums) # 3
    assert params.size(1) == sum(weight_nums) + sum(bias_nums) # 169

    num_insts = params.size(0) # 160
    num_layers = len(weight_nums) # 3

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))  # 6

    weight_splits = params_splits[:num_layers]   # 3
    bias_splits = params_splits[num_layers:]   # 3

    for l in range(num_layers):  # 3
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits  # 见下方注释


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):

        # 设置好了参数num_gen_params
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS  # 3
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS    # 8
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS   # 8
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE   # 4
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS   # False

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST  # size of interest [64, 128, 256, 512] focal的参数 就是每一层中max(l, r, t, b)
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS

        weight_nums, bias_nums = [], []   # weights, bias个数
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)  # 8 + 2 = 10 加入rel coord
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)  #  8
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums  # [80, 64, 8]
        self.bias_nums = bias_nums   # [8, 8, 1]
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)   # 169

        self.register_buffer("_iter", torch.zeros([1]))

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
    # mask_feats torch.Size([2, 8, 100, 152])
    # mask_feat_stride = 8
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )   # [15200, 2]
        n_inst = len(instances)

        im_inds = instances.im_inds  # 160  160为此次训练的这样本总个数 下同
        mask_head_params = instances.mask_head_params   # [160, 169]

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations   # [160, 2]
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)   # [160, 1, 2] - [1, 15200, 2] = [160, 15200, 2]
            relative_coords = relative_coords.permute(0, 2, 1).float()  # [160, 2, 15200]
            soi = self.sizes_of_interest.float()[instances.fpn_levels]   # [64] 下方注释 存储了映射的stride
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)   # soi.reshape(-1, 1, 1) --> [160, 1 ,1]  为什么要除以Soi 如何理解？
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)    # torch.Size([160, 2, 15200])

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)   # torch.Size([160, 10, 15200])
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)   # torch.Size([1, 1600, 100, 152])

        weights, biases = parse_dynamic_params(   # 调用parse_dynamic_params 见下方注释
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)   # torch.Size([160, 1, 100, 152])

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))  # 插值 torch.Size([160, 1, 200, 304])

        return mask_logits   # sigmoid

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):  # eg. torch.Size([2, 8, 100, 152])  8  160个instnaces 2个gt_instances  gt_instances[0] = 15 gt_instances[1] = 3
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])  # 循环batchsize次 gt[0] : [15, 200, 304] gt[1] : [3, 200, 304]
            # 根据索引[160]里的数字是 0-17(见下方注释)来筛选原来gt_bitmasks的某维度(gt_inds[0] = 0 就对于第0维的值),添加到160的维度。
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)  # [160, 1, 200, 304]

            losses = {}

            if len(pred_instances) == 0:   # 160
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                if not self.boxinst_enabled:
                    losses["loss_mask"] = dummy_loss
                else:
                    losses["loss_prj"] = dummy_loss
                    losses["loss_pairwise"] = dummy_loss
            else:
                mask_logits = self.mask_heads_forward_with_coords(    # 调用mask_heads_forward_with_coords 得到mask_scores
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_scores = mask_logits.sigmoid()

                if self.boxinst_enabled:
                    # box-supervised BoxInst losses
                    image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                    image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)  #[160] 维度的loss

                    pairwise_losses = compute_pairwise_term(
                        mask_logits, self.pairwise_size,
                        self.pairwise_dilation
                    )

                    weights = (image_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
                    loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

                    warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                    loss_pairwise = loss_pairwise * warmup_factor

                    losses.update({
                        "loss_prj": loss_prj_term,
                        "loss_pairwise": loss_pairwise,
                    })
                else:
                    # fully-supervised CondInst losses
                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                    losses["loss_mask"] = loss_mask

            return losses
        else:
            if len(pred_instances) > 0:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()

            return pred_instances

'''
1. gt_bitmasks
gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances]) # 循环batchsize次
    (Pdb) gt_instances[0].gt_bitmasks.size()
    torch.Size([15, 200, 304])
    (Pdb) gt_instances[1].gt_bitmasks.size()
    torch.Size([3, 200, 304])

2. gt_bitmasks
 gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
 [160, 1, 200, 304]


(Pdb) pred_instances.gt_inds

tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  6,  6,  5,  5,  5,  6,  6,  6,
         5,  5,  5,  6,  6,  6,  9,  9,  9,  8,  8,  8, 12, 12,  5, 10, 10, 10,
        11, 11,  4,  4,  4,  9,  9,  9,  8,  8,  8, 12, 12, 10, 10, 10, 13, 13,
        11, 11,  4,  4,  4,  9,  9,  9,  8,  8,  8, 12, 12, 10, 10, 10, 13, 13,
        11, 11,  4,  4,  4, 17, 17, 17, 17, 17, 17, 17, 17, 17,  1,  1,  1,  3,
         3,  1,  1,  1,  3,  3,  2,  2,  2,  1,  1,  1,  3,  3,  2,  2,  2,  4,
         4,  4,  2,  2,  2,  4,  4,  4, 14, 14, 14, 14, 14, 14, 14, 14, 14, 17,
        17, 17, 15, 15, 15,  2,  2,  7,  7,  7,  7,  7,  7,  7,  7,  7, 14, 14,
        15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
       device='cuda:0')

(Pdb) soi
tensor([  64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,
          64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,
          64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,
          64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,
          64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,
          64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,
          64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,
          64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,   64.,
          64.,   64.,   64.,   64.,   64.,   64.,  128.,  128.,  128.,  128.,
         128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,
         128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,
         128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,
         128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,  128.,
         128.,  256.,  256.,  256.,  256.,  256.,  256.,  256.,  256.,  256.,
         256.,  256.,  256.,  256.,  256.,  256.,  256.,  256.,  256.,  256.,
         512.,  512.,  512., 1024., 1024., 1024., 1024., 1024., 1024., 1024.],
       device='cuda:0')
(Pdb) soi.size()
torch.Size([160])


(Pdb) instances.fpn_levels
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4], device='cuda:0')


(Pdb) mask_head_inputs.size()
torch.Size([1, 1600, 100, 152])
(Pdb)  self.channels
8
(Pdb)  self.bias_nums
[8, 8, 1]
(Pdb) self.weight_nums
[80, 64, 8]
(Pdb) mask_head_params.size()
torch.Size([160, 169])


parse_dynamic_param()方法

(Pdb) len(weight_splits)
3
(Pdb) weight_splits[0].size()
torch.Size([1280, 10, 1, 1])
(Pdb) weight_splits[1].size()
torch.Size([1280, 8, 1, 1])
(Pdb) weight_splits[2].size()
torch.Size([160, 8, 1, 1])



(Pdb) len(bias_splits)
3
(Pdb) bias_splits[0].size()
torch.Size([1280])
(Pdb) bias_splits[1].size()
torch.Size([1280])
(Pdb) bias_splits[2].size()
torch.Size([160])

'''
