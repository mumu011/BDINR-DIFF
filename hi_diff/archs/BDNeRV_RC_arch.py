from hi_diff.utils.bd_modules import Conv, ResBlock, MLP, CUnet
import torch.nn as nn
import torch
from hi_diff.utils.bd_utils import PositionalEncoding
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY

class GELU_MLP(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 **kwargs):

        super(GELU_MLP, self).__init__()

        patch_mix_dims = int(patch_expansion * num_patches)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Linear(patch_mix_dims, num_patches),
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Linear(channel_mix_dims, embed_dims),
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.patch_mixer(self.norm1(x).transpose(1,2)).transpose(1,2)
        x = x + self.channel_mixer(self.norm2(x))

        return x

@ARCH_REGISTRY.register()
class BDNeRV_RC(nn.Module):
    # recursive frame reconstruction
    def __init__(self, n_colors=3, n_resblock=4, n_feats=32, kernel_size=3, padding=1, group=8, patch_expansion=0.5, channel_expansion=4):
        super(BDNeRV_RC, self).__init__()

        pos_b, pos_l = 1.25, 80  # position encoding params
        mlp_dim_list = [2*pos_l, 512, n_feats*4*2] # (160, 512, 256)
        mlp_act = 'gelu'

        # main body
        self.mainbody = CUnet(n_feats=n_feats, n_resblock=n_resblock,
                              kernel_size=kernel_size, padding=padding)

        # # output block
        # OutBlock = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
        #             ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
        #             Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        # self.out = nn.Sequential(*OutBlock)

        # feature block
        FeatureBlock = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature = nn.Sequential(*FeatureBlock)

        # feature block gt
        FeatureBlock_gt = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature_gt = nn.Sequential(*FeatureBlock_gt)

        # concatenation fusion block
        CatFusion = [Conv(input_channels=n_feats*2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.catfusion = nn.Sequential(*CatFusion)

        # concatenation fusion block gt
        CatFusion_gt = [Conv(input_channels=n_feats*2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.catfusion_gt = nn.Sequential(*CatFusion_gt)

        # position encoding
        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)

        # mlp
        self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)

        self.pool = nn.AdaptiveAvgPool2d((group, group))
        self.mlp = GELU_MLP(num_patches=group*group, embed_dims=n_feats, patch_expansion=patch_expansion, channel_expansion=channel_expansion)
        self.end = nn.Sequential(
                nn.Linear(n_feats, n_feats*4),
                nn.GELU(),)

    def forward(self, ce_blur, time_idx, ce_code, gt=None):
        # time index: [frame_num,1]
        # t_embed
        t_pe_ = [self.pe_t(idx)*(2*code-1)
                 for idx, code in zip(time_idx, ce_code)]  # [frame_num*[pos_l*2,1]]
        t_pe = torch.cat(t_pe_, dim=0)  # [frame_num, pos_l*2]
        t_embed = self.embed_mlp(t_pe)  # [frame_num, n_feats*4*2]
        # t_manip = self.manip_mlp(t_pe)

        if gt is not None:
            gt_list = torch.split(gt, 1, dim=1)

        # ce_blur feature
        ce_feature = self.feature(ce_blur)  # [b, 32, h, w]

        # main body
        # output_list = []
        feat_list = []
        for k in range(len(time_idx)):
            if k==0:
                main_feature = ce_feature
            else:
                # since k=2, cat pre-feature with ce_feature as input feature
                cat_feature = torch.cat((feat_out_k, ce_feature),dim=1)
                main_feature = self.catfusion(cat_feature)
            # # non-recursive
            # main_feature = ce_feature
            feat_out_k = self.mainbody(main_feature, t_embed[k])  # [b, 32, h, w]
            if gt is not None:
                feat_out_k_gt = self.feature_gt(gt_list[k].squeeze(1))
                feat_out_k = self.catfusion_gt(torch.cat((feat_out_k, feat_out_k_gt), dim=1))
            feat_out_c = self.pool(feat_out_k)  # [b, 32, group, group]
            feat_out_c = rearrange(feat_out_c, 'b c h w-> b (h w) c') # [b, group*group, 32]
            feat_out_c = self.mlp(feat_out_c)   # [b, group*group, 32]
            feat_out_c = self.end(feat_out_c)   # [b, group*group, 128]
            feat_list.append(feat_out_c)
            # output_k = self.out(feat_out_k)  # [b, 3, h, w]
            # output_list.append(output_k)
            # output_k = self.out(feat_out_k)
            # feat_list.append(output_k)

        # output = torch.stack(output_list, dim=1)

        return feat_list
