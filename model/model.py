import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
import logging
import torch.distributed as dist
import numpy as np
from clip import clip
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from knn_cuda import KNN

# NOTE: some parts are adopted from Point-M2AE code base (https://github.com/ZrrSkywalker/Point-M2AE)


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)
    
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



logger_initialized = {}

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True


    return logger

def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')

class Token_Embed(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        if in_c == 3:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, out_c, 1)
            )

        else:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, in_c, 1),
                nn.BatchNorm1d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_c, in_c, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(in_c * 2, out_c, 1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_c, out_c, 1)
            )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.out_c)
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # mask = mask * float('-inf') 
            mask = mask * - 100000.0
            attn = attn + mask.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # global flag
        # flag += 1
        # if flag == 5:
        #     for k in range(attn.shape[0]):
        #         torch.save(attn[k][0][0][:], "/data2/renrui/visualize_pc/layer5_mask/data/attn" + str(k) + ".pt")
        #     exit(1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class Encoder_Block(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()    
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos, vis_mask):
        for _, block in enumerate(self.blocks):
            x = block(x + pos, vis_mask)
        return x
    

# Hierarchical Encoder
class H_Encoder_seg(nn.Module):

    def __init__(self, encoder_depths=[5, 5, 5], num_heads=6, encoder_dims=[96, 192, 384], local_radius=[0.32, 0.64, 1.28]):
        super().__init__()

        self.encoder_depths = encoder_depths
        self.encoder_num_heads = num_heads
        self.encoder_dims = encoder_dims
        self.local_radius = local_radius

        # token merging and positional embeddings
        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embed.append(Token_Embed(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.token_embed.append(Token_Embed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))
            
            self.encoder_pos_embeds.append(nn.Sequential(
                            nn.Linear(3, self.encoder_dims[i]),
                            nn.GELU(),
                            nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
                        ))

        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(Encoder_Block(
                            embed_dim=self.encoder_dims[i],
                            depth=self.encoder_depths[i],
                            drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                            num_heads=self.encoder_num_heads,
                        ))
            depth_count += self.encoder_depths[i]

        self.encoder_norms = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.encoder_norms.append(nn.LayerNorm(self.encoder_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs, eval=False):
        # hierarchical encoding
        x_vis_list = []
        xyz_dist = None
        for i in range(len(centers)):
            # 1st-layer encoder, conduct token embedding
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            # intermediate layers, conduct token merging
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)
            
            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(centers[i], self.local_radius[i], xyz_dist)
                mask_vis_att = mask_radius 
            else:
                mask_vis_att = None

            pos = self.encoder_pos_embeds[i](centers[i])
            x_vis = self.encoder_blocks[i](group_input_tokens, pos, mask_vis_att)
            x_vis_list.append(x_vis)

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i]).transpose(-1, -2).contiguous()
        return x_vis_list

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, idx

class PointNetFeaturePropagation_(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation_, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2, is_learned=True, use_ori_pc=True):
        """
        Input:
            xyz1: input points position data, [B, C, N]; [BS=16, xyz=3, n_pts=2048] --> original pc
            xyz2: sampled input points position data, [B, C, S]; [BS=16, xyz=3, n_center=512] --> sampled center
            points1: input points data, [B, D, N]; [BS=16, xyz=3, n_pts=2048] --> original pc
            points2: input points data, [B, D, S]; [BS=16, , e=96, n_center=512] --> features
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None and use_ori_pc:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1) #[BS=16, n_pts=2048, e=99]
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        if is_learned:
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
        return new_points
    
class Point_M2AE_SEG(nn.Module):
    def __init__(self, num_classes, is_learned, text_prompts=None):
        super().__init__()

        assert text_prompts != None, "Text prompt can be None!"
        
        self.trans_dim = 384
        self.group_sizes = [16, 8, 8]
        self.num_groups = [512, 256, 64]
        self.num_classes = num_classes
        self.encoder_dims = [96, 192, 384]

        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder_seg()
       
        # to output encoded text
        clip_model, _ = clip.load('ViT-B/16')
        clip_model = clip_model.eval()
        text_prompts_united = [" ".join(text_prompts)]
        self.text_feat = self.text_encoder(clip_model, text_prompts=text_prompts_united)
        self.text_feat = self.text_feat.detach().type(torch.float32)
        self.text_feat = self.text_feat / self.text_feat.norm(dim=-1, keepdim=True)
        self.label_conv_pre = nn.Sequential(nn.Conv1d(512, 16, kernel_size=1, bias=False),
                                nn.BatchNorm1d(16),
                                nn.LeakyReLU(0.2)) 
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2)) # 

        self.propagations = nn.ModuleList()
        for i in range(3):
            self.propagations.append(PointNetFeaturePropagation_(in_channel=self.encoder_dims[i] + 3, mlp=[self.trans_dim * 4, 1024]))

        self.convs1 = nn.Conv1d(6208, 1024, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(1024, 512, 1)
        self.convs3 = nn.Conv1d(512, 256, 1)
        self.convs4 = nn.Conv1d(256, self.num_classes, 1)
        self.bns1 = nn.BatchNorm1d(1024)
        self.bns2 = nn.BatchNorm1d(512)
        self.bns3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.is_learned = is_learned

    def text_encoder(self, clip_model, text_prompts):
        prompts = torch.cat([clip.tokenize(p) for p in text_prompts]).cuda() #[n_parts=4, context_length=77]
        text_feat = clip_model.encode_text(prompts) #[n_parts=, e=512]
        return text_feat
    
    def load_model_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        incompatible = self.load_state_dict(state_dict['base_model'], strict=False)
        if incompatible.missing_keys:
            # print_log('missing_keys', logger='Point_M2AE_ModelNet40')
            pass
           
        if incompatible.unexpected_keys:
            # print_log('unexpected_keys', logger='Point_M2AE_ModelNet40')
            pass
        print("Point-M2AE is loaded!")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, use_ori_pc=True):

        B, C, N = x.shape
        x = x.transpose(-1, -2).contiguous() # B N 3
        # divide the point cloud in the same form. This is important
        
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](x)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # b*g*k

        # hierarchical encoder
        x_vis_list = self.h_encoder(neighborhoods, centers, idxs, eval=True) #[BS=16, e=96, n_pts=512], [BS=16, e=192, n_pts=256], [BS=16, e=384, n_pts=64]

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.propagations[i](x.transpose(-1, -2), centers[i].transpose(-1, -2), x.transpose(-1, -2), x_vis_list[i], is_learned=self.is_learned, use_ori_pc=use_ori_pc) # list of [16, 1024, 2048], [16, 1024, 2048] & [16, 1024, 2048]
            
        x = torch.cat((x_vis_list[0], x_vis_list[1], x_vis_list[2]), dim=1)  # [BS=16, 3072, 2048] if not is_learned, else [BS=16, 681=96+3+192+3+384+3, 2048], or [BS=16, 672=96+192+384, 2048]
        pretrained_feat = torch.clone(x)

        if not self.is_learned:
            return pretrained_feat #[BS=16, e=681, n_pts=2048]

        x_max = torch.max(x, 2)[0] #[BS=16, e=3072]
        x_avg = torch.mean(x, 2) #[BS=16, e=3072]
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        
        if x.shape[0] > self.text_feat.shape[0]:
            text_feat = torch.tile(self.text_feat, (x.shape[0],1))
        conditioned_token = self.label_conv_pre(text_feat.unsqueeze(dim=-1)) # [BS=8, e=512, 1] --> [BS=8, e=16, 1]
        conditioned_token = self.label_conv(conditioned_token).repeat(1, 1, N) # [BS=8, e=16, 1] --> [BS=8, e=64, n_pts]
        
        x_global_feature = torch.cat((x_max_feature + x_avg_feature, conditioned_token), 1) # [BS=16, e=3136=3072+64, n_pts=2048])

        x = torch.cat((x_global_feature, x), 1) #[BS=16, e=6208, n_pts=2048]
        x = self.relu(self.bns1(self.convs1(x))) #[BS=16, e=1024, n_pts=2048]
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x))) #[BS=16, e=512, n_pts=2048]
        x = self.relu(self.bns3(self.convs3(x))) #[BS=16, e=256, n_pts=2048]
        x = self.convs4(x) #[BS=16, e=50, n_pts=2048]
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1) #[BS=16, n_pts=2048, n_part=5]
        return x


class StudentNetworks(nn.Module):
    def __init__(self, args, num_classes, text_prompts=None, emb=681):
        super().__init__()
        
        self.num_classes = num_classes
        
        # define the extractor
        self.extractor = Point_M2AE_SEG(num_classes=num_classes, text_prompts=text_prompts, is_learned=False)
        
        # load Point-M2AE ckpt
        state_dict = torch.load(args.backbone_path)
        incompatible = self.extractor.load_state_dict(state_dict['base_model'], strict=False)
        if incompatible.missing_keys:
            pass
        if incompatible.unexpected_keys:
            pass
        self.freeze_extractor()
        print("Point-M2AE is loaded!")
        
        self.fc1 = nn.Linear(emb, emb)
        self.fc2 = nn.Linear(emb, emb)
        self.fc3 = nn.Linear(emb, emb)
        self.fc4 = nn.Linear(emb, num_classes)

    
    def text_encoder(self, clip_model, text_prompts):
        prompts = torch.cat([clip.tokenize(p) for p in text_prompts]).cuda() #[n_parts=4, context_length=77]
        text_feat = clip_model.encode_text(prompts) #[n_parts=, e=512]
        return text_feat
    
    def distillation_head(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def freeze_extractor(self):
        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self, x, return_pts_feat=False):
        feature = self.extractor(x) #[BS, e=681, n_pts=2048]
        
        x = self.distillation_head(feature.permute(0,2,1))
        
        if return_pts_feat:
            return x, feature
        else:     
            return x