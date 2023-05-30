import torch
import torchvision.transforms as transforms

from .. import nn


def nonlinearity(x):
  #return tf.nn.swish(x)
  return torch.nn.functional.silu(x)


def normalize(x, *, temb):
  # return tf_contrib.layers.group_norm(x, scope=name)
  return nn.GroupNorm(1, x.size()[1])(x)


# def upsample(x, *, name, with_conv):
#   with tf.variable_scope(name):
#     B, H, W, C = x.shape
#     x = tf.image.resize(x, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
#     assert x.shape == [B, H * 2, W * 2, C]
#     if with_conv:
#       x = nn.conv2d(x, name='conv', num_units=C, filter_size=3, stride=1)
#       assert x.shape == [B, H * 2, W * 2, C]
#     return x

class upsample(torch.nn.Module):
  def __init__(self, with_conv, C):
    self.with_conv = with_conv
    if self.with_conv:
      self.conv = torch.nn.LazyConv2d(out_channels=C, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)

  def forward(self, x):
    B, H, W, C = x.shape
    x = torch.nn.functional.interpolate(x, size=(H * 2, W * 2), mode='nearest', align_corners=True)
    if self.with_conv:
      x = self.conv(x)
    return x



# def downsample(x, *, name, with_conv):
#   with tf.variable_scope(name):
#     B, H, W, C = x.shape
#     if with_conv:
#       x = nn.conv2d(x, name='conv', num_units=C, filter_size=3, stride=2)
#     else:
#       x = tf.nn.avg_pool(x, 2, 2, 'SAME')
#     assert x.shape == [B, H // 2, W // 2, C]
#     return x
  
class downsample(torch.nn.Module):
  def __init__(self, with_conv, C):
    self.with_conv = with_conv
    if self.with_conv:
      self.conv = torch.nn.LazyConv2d(out_channels=C, kernel_size=3, stride=2, dilation=1, padding='same', bias=True)

  def forward(self, x):
    B, H, W, C = x.shape
    if self.with_conv:
      x = self.conv(x)
    else:
      x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2, padding=0, ceil_mode=False)
    return x


# def resnet_block(x, *, temb, name, out_ch=None, conv_shortcut=False, dropout):
#   B, H, W, C = x.shape
#   if out_ch is None:
#     out_ch = C

#   with tf.variable_scope(name):
#     h = x

#     h = nonlinearity(normalize(h, temb=temb, name='norm1'))
#     h = nn.conv2d(h, name='conv1', num_units=out_ch)

#     # add in timestep embedding
#     h += nn.dense(nonlinearity(temb), name='temb_proj', num_units=out_ch)[:, None, None, :]

#     h = nonlinearity(normalize(h, temb=temb, name='norm2'))
#     h = tf.nn.dropout(h, rate=dropout)
#     h = nn.conv2d(h, name='conv2', num_units=out_ch, init_scale=0.)

#     if C != out_ch:
#       if conv_shortcut:
#         x = nn.conv2d(x, name='conv_shortcut', num_units=out_ch)
#       else:
#         x = nn.nin(x, name='nin_shortcut', num_units=out_ch)

#     assert x.shape == h.shape
#     print('{}: x={} temb={}'.format(tf.get_default_graph().get_name_scope(), x.shape, temb.shape))
#     return x + h
  
class Resnet_block(torch.nn.Module):
  def __init__(self, out_ch, dropout = 0.):
    self.dense_1 = torch.nn.LazyLinear(out_ch)
    self.conv_1 = torch.nn.LazyConv2d(out_channels=out_ch, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)
    self.conv_2 = torch.nn.LazyConv2d(out_channels=out_ch, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)
    self.conv_3 = torch.nn.LazyConv2d(out_channels=out_ch, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)
    
    self.dropout = dropout

  def forward(self, x, *, temb, out_ch=None, conv_shortcut=False, dropout):
    B, H, W, C = x.shape
    self.dropout = dropout
    if out_ch is None:
      out_ch = C

    h = x

    h = nonlinearity(normalize(h))
    h = self.conv_1(h)

    # add in timestep embedding
    h = h + self.dense_1((nonlinearity(temb)))[:, None, None, :]

    h = nonlinearity(normalize(h))
    h = torch.nn.functional.dropout(h, p=self.dropout)
    h = self.conv_2(h)

    if C != out_ch:
      if conv_shortcut:
        x = self.conv_3(x)
      else:
        x = nn.nin(x, name='nin_shortcut', num_units=out_ch)##nin ?

    return x + h
  

# def attn_block(x, *, name, temb):
#   B, H, W, C = x.shape
#   with tf.variable_scope(name):
#     h = normalize(x, temb=temb, name='norm')
#     q = nn.nin(h, name='q', num_units=C)
#     k = nn.nin(h, name='k', num_units=C)
#     v = nn.nin(h, name='v', num_units=C)

#     w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
#     w = tf.reshape(w, [B, H, W, H * W])
#     w = tf.nn.softmax(w, -1)
#     w = tf.reshape(w, [B, H, W, H, W])

#     h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
#     h = nn.nin(h, name='proj_out', num_units=C, init_scale=0.)

#     assert h.shape == x.shape
#     print(tf.get_default_graph().get_name_scope(), x.shape)
#     return x + h

class Attn_block(torch.nn.Module):
  def __init__(self, C):
    self.q_dense = torch.nn.LazyLinear(out_features=C)
    self.k_dense = torch.nn.LazyLinear(out_features=C)
    self.v_dense = torch.nn.LazyLinear(out_features=C)
    self.out_dense = torch.nn.LazyLinear(out_features=C)

  def forward(self, x, temb):
    B, H, W, C = x.shape
    
    h = normalize(x, temb=temb, name='norm')
    q = self.q_dense(h)
    k = self.k_dense(h)
    v = self.v_dense(h)

    w = torch.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = w.view([B, H, W, H * W])
    w = torch.nn.functional.softmax(w, dim = -1)
    w = w.view([B, H, W, H, W])

    h = torch.einsum('bhwHW,bHWc->bhwc', w, v)
    h = self.out_dense(h)

    return x + h

# def model(x, *, t, y, name, num_classes, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
#           attn_resolutions, dropout=0., resamp_with_conv=True):
#   B, S, _, _ = x.shape
#   assert x.dtype == tf.float32 and x.shape[2] == S
#   assert t.dtype in [tf.int32, tf.int64]
#   num_resolutions = len(ch_mult)

#   assert num_classes == 1 and y is None, 'not supported'
#   del y

#   with tf.variable_scope(name, reuse=reuse):
#     # Timestep embedding
#     with tf.variable_scope('temb'):
#       temb = nn.get_timestep_embedding(t, ch)
#       temb = nn.dense(temb, name='dense0', num_units=ch * 4)
#       temb = nn.dense(nonlinearity(temb), name='dense1', num_units=ch * 4)
#       assert temb.shape == [B, ch * 4]

#     # Downsampling
#     hs = [nn.conv2d(x, name='conv_in', num_units=ch)]
#     for i_level in range(num_resolutions):
#       with tf.variable_scope('down_{}'.format(i_level)):
#         # Residual blocks for this resolution
#         for i_block in range(num_res_blocks):
#           h = resnet_block(
#             hs[-1], name='block_{}'.format(i_block), temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
#           if h.shape[1] in attn_resolutions:
#             h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
#           hs.append(h)
#         # Downsample
#         if i_level != num_resolutions - 1:
#           hs.append(downsample(hs[-1], name='downsample', with_conv=resamp_with_conv))

#     # Middle
#     with tf.variable_scope('mid'):
#       h = hs[-1]
#       h = resnet_block(h, temb=temb, name='block_1', dropout=dropout)
#       h = attn_block(h, name='attn_1'.format(i_block), temb=temb)
#       h = resnet_block(h, temb=temb, name='block_2', dropout=dropout)

#     # Upsampling
#     for i_level in reversed(range(num_resolutions)):
#       with tf.variable_scope('up_{}'.format(i_level)):
#         # Residual blocks for this resolution
#         for i_block in range(num_res_blocks + 1):
#           h = resnet_block(tf.concat([h, hs.pop()], axis=-1), name='block_{}'.format(i_block),
#                            temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
#           if h.shape[1] in attn_resolutions:
#             h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
#         # Upsample
#         if i_level != 0:
#           h = upsample(h, name='upsample', with_conv=resamp_with_conv)
#     assert not hs

#     # End
#     h = nonlinearity(normalize(h, temb=temb, name='norm_out'))
#     h = nn.conv2d(h, name='conv_out', num_units=out_ch, init_scale=0.)
#     assert h.shape == x.shape[:3] + [out_ch]
#     return h


class UNET(torch.nn.Module):
  def __init__(self, ch, out_ch, num_res_blocks, attn_resolutions, ch_mult=(1, 2, 4, 8), resamp_with_conv=True):

    self.ch = ch
    self.out_ch = out_ch
    self.num_res_blocks = num_res_blocks
    self.ch_mult = ch_mult
    self.attn_resolutions = attn_resolutions

    self.temb_dense_1 = torch.nn.LazyLinear(ch * 4)
    self.temb_dense_2 = torch.nn.LazyLinear(ch * 4)

    self.conv_1 = torch.nn.LazyConv2d(out_channels=ch, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)
    self.conv_2 = torch.nn.LazyConv2d(out_channels=out_ch, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)

    self.down_resnet = []
    self.down_attention = []
    for i in range(len(ch_mult)):
      res = []
      att = []
      for j in range(num_res_blocks):
        res.append(Resnet_block(ch))
        att.append(Attn_block(ch))
      self.down_resnet.append(res)
      self.down_attention.append(att)
    
    self.up_resnet = []
    self.up_attention = []
    for i in range(len(ch_mult)):
      res = []
      att = []
      for j in range(num_res_blocks):
        res.append(Resnet_block(ch))
        att.append(Attn_block(ch))
      self.up_resnet.append(res)
      self.up_attention.append(att)

    self.downsample = []
    self.upsample = []
    for i in range(len(ch_mult)-1):
      self.downsample.append(downsample(resamp_with_conv, ch))
      self.upsample.append(upsample(resamp_with_conv, ch))


  def forward(self, x, *, t):
    
    num_resolutions = len(self.ch_mult)
    
    temb = nn.get_timestep_embedding(t, self.ch)
    temb = nonlinearity(self.temb_dense_1(temb))
    temb = self.temb_dense_2(temb)

    hs = [self.conv_1(x)]
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = self.down_resnet[i_level][i_block](hs[-1])
        if h.shape[1] in self.attn_resolutions:
          h = self.down_attention[i_level][i_block](h)
        hs.append(h)
      # Downsample
      if i_level != num_resolutions - 1:
        hs.append(self.downsample[i_level](hs[-1]))

    
    h = hs[-1]
    h = self.resnet2(h)
    h = self.attn_block_2(h)
    h = self.resnet3(h)

  # Upsampling
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = self.up_resnet[i_level][i_block](torch.cat([h, hs.pop()], dim=-1))
        if h.shape[1] in self.attn_resolutions:
          h = self.up_attention[i_level][i_block](h)
      # Upsample
      if i_level != 0:
        h = self.upsample[i_level-1](h)
    # assert not hs

    # End
    h = nonlinearity(normalize(h))
    h = self.conv_2(h)
    # assert h.shape == x.shape[:3] + [out_ch]

    return h

