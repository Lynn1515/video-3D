import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, Tuple, List, Callable, Dict

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

from lvdm.common import (
    checkpoint,
    exists,
    default,
)

from torchvision.utils import save_image
from einops import rearrange, repeat


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.use_efficient = False # use xformers memory efficient attention if available

    def after_step(self):
        pass

    # def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
    #     out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
    #     self.cur_att_layer += 1
    #     if self.cur_att_layer == self.num_att_layers:
    #         self.cur_att_layer = 0
    #         self.cur_step += 1
    #         # after step
    #         self.after_step()
    #     return out

    def __call__(self, q, k, v, *args, k_ip=None, v_ip=None, is_cross=False, place_in_unet=None, num_heads=None, b=None, efficient=False, **kwargs):
        if efficient:
            out, out_ip = self.efficient_forward(q, k, v, k_ip=k_ip, v_ip=v_ip, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, b=b, **kwargs)
        else:
            # assume sim and attn are in args or kwargs
            if len(args) >= 2:
                sim, attn = args[:2]
            else:
                sim, attn = kwargs.get('sim'), kwargs.get('attn')
            out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.after_step()
        return out, out_ip

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def efficient_forward(self, q, k, v, k_ip, v_ip, is_cross, place_in_unet, num_heads, **kwargs):
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        out_ip = None
        if k_ip is not None:
            out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=num_heads)

        return out, out_ip

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]
        self.self_attns_step.clear()
        self.cross_attns_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)


def regiter_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


def regiter_attention_editor_ldm(model, editor: AttentionBase):
    """
    Register a attention editor to Stable Diffusion model, refer from [Prompt-to-Prompt]
    """

    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale, b=2)

            return to_out(out)

        return forward

    def ca_forward_efficent(self, place_in_unet, efficient=True):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            is_cross = context is not None
            h = self.heads
            q = self.to_q(x)
            #b, n, _ = x.shape
            spatial_self_attn = context is None
            context = default(context, x)
            
            #is_cross = context is not None
            #context = context if is_cross else x
            # project to q, k, v
            k_ip, v_ip, out_ip = None, None, None
            
            if self.image_cross_attention and not spatial_self_attn:#(50,77,1024)#(50,256,1024)
                context, context_image = context[:, :self.text_context_len, :], context[:, self.text_context_len:, :]
                k = self.to_k(context)
                v = self.to_v(context)#(50,77,320)
                k_ip = self.to_k_ip(context_image)#(50,256,320)
                v_ip = self.to_v_ip(context_image)
            else:
                if not spatial_self_attn:
                    context = context[:, :self.text_context_len, :]
                k = self.to_k(context)
                v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            if k_ip is not None:
                k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            # q = self.to_q(x)#(25,9216,320)
            # k = self.to_k(context)
            # v = self.to_v(context)
            # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            if self.relative_position or self.temporal_length is not None:
                # do relative position embedding when using Temporal Transformer
                assert(self.temporal_length is not None)#temporal modeling not efficent
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
                
                if self.relative_position:
                    len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
                    k2 = self.relative_position_k(len_q, len_k)
                    sim2 = torch.einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
                    sim += sim2
                del k
                # attention, what we cannot get enough of
                sim = sim.softmax(dim=-1)

                out = torch.einsum('b i j, b j d -> b i d', sim, v)
                if self.relative_position:
                    v2 = self.relative_position_v(len_q, len_v)
                    out2 = torch.einsum('b t s, t s d -> b t d', sim, v2) # TODO check
                    out += out2
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)


                ## for image cross-attention
                if k_ip is not None:
                    #k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
                    sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
                    del k_ip
                    sim_ip = sim_ip.softmax(dim=-1)
                    out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
                    out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)


            else:   #spatial transformer
                if XFORMERS_IS_AVAILBLE and self.temporal_length is None:
                    # use memory-efficient attention via xformers
                    out, out_ip = editor(
                        q, k, v, None, None, k_ip=k_ip, v_ip=v_ip, is_cross=is_cross, place_in_unet=place_in_unet,
                        num_heads=self.heads, b=2, efficient=True, scale=self.scale
                    )


            if out_ip is not None:
                if self.image_cross_attention_scale_learnable:
                    out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha)+1)
                else:
                    out = out + self.image_cross_attention_scale * out_ip
        
            return to_out(out)

        return forward
    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'CrossAttention':  # spatial Transformer layer
                net.forward = ca_forward_efficent(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.model.diffusion_model.named_children():
        if "input" in net_name:
            cross_att_count += register_editor(net, 0, "input")
        elif "middle" in net_name:
            cross_att_count += register_editor(net, 0, "middle")
        elif "output" in net_name:
            cross_att_count += register_editor(net, 0, "output")
    editor.num_att_layers = cross_att_count
    print("total cross_attn count", cross_att_count)
