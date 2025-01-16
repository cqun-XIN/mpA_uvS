# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class STA_SelfAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, freq=10000, heads=1, pos_enc=None, num_segments=4):
        """ The basic (multi-head) Attention 'cell' containing the learnable parameters of Q, K and V

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param int heads: Number of heads for the attention module.
        :param str | None pos_enc: The type of the positional encoding [supported: Absolute, Relative].
        """
        super(STA_SelfAttention, self).__init__()

        self.permitted_encodings = ["absolute", "relative"]
        if pos_enc is not None:
            pos_enc = pos_enc.lower()
            #assert断言。https://blog.csdn.net/guhuoone/article/details/124540721
            assert pos_enc in self.permitted_encodings, f"Supported encodings: {*self.permitted_encodings,}"

        self.input_size = input_size
        self.output_size = output_size
        self.heads = heads
        self.pos_enc = pos_enc
        self.freq = freq
        self.scale = num_segments
        '''nn.ModuleList()类似于n.Sequential()，但不相同。https://zhuanlan.zhihu.com/p/75206669'''
        self.Wk, self.Wq, self.Wv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(self.heads):
            '''output_size//heads：表示向下取整的除法。https://blog.csdn.net/SummerMiko_/article/details/123428734'''
            self.Wk.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
            self.Wq.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
            self.Wv.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
        self.out = nn.Linear(in_features=output_size, out_features=input_size, bias=False)
        self.out_2 = nn.Linear(in_features=(output_size // heads // num_segments) + 2, out_features=output_size // heads // num_segments, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=0.5)

    def getAbsolutePosition(self, T, num):
        """Calculate the sinusoidal positional encoding based on the absolute position of each considered frame.
        Based on 'Attention is all you need' paper (https://arxiv.org/abs/1706.03762)

        :param int T: Number of frames contained in Q, K and V
        :return: Tensor with shape [T, T]
        """
        freq = self.freq
        d = self.input_size
        m = num

        pos = []
        #i = []
        for _ in range(m):
            pos_tmp = torch.tensor([k for k in range(T)], device=self.out.weight.device)
            #i_tmp = torch.tensor([k for k in range(T // 2)], device=self.out.weight.device)
            pos.append(pos_tmp)
            #i.append(i_tmp)
        pos = torch.stack(pos, dim=0)
        #i = torch.stack(i, dim=0)
        i = torch.tensor([k for k in range(T // 2)], device=self.out.weight.device)

        # Reshape tensors each pos_k for each i indices
        pos = pos.reshape(m,pos.shape[1], 1)
        '''
           repeat_interleave是将张量中的元素沿某一维度复制n次，即复制后的张量沿该维度相邻的n个元素是相同的。
           repeat_interleave()中有两个参数，第一个参数N是代表复制多少次，第二个参数代表维度。
           https://blog.csdn.net/starlet_kiss/article/details/125718922
        '''
        pos = pos.repeat_interleave(i.shape[0], dim=2)
        '''复制的一种方式，与repeat_interleave略有不同。https://blog.csdn.net/starlet_kiss/article/details/125718922'''
        i = i.repeat(m, pos.shape[1], 1)

        AP = torch.zeros(m, T, T, device=self.out.weight.device)
        for j in range(m):
            AP[j, pos, 2 * i] = torch.sin(pos / freq ** ((2 * i) / d))
            AP[j, pos, 2 * i + 1] = torch.cos(pos / freq ** ((2 * i) / d))
        # AP[pos, 2*i] = torch.sin(pos / freq ** ((2 * i) / d))
        # AP[pos, 2*i+1] = torch.cos(pos / freq ** ((2 * i) / d))
        return AP

    def getRelativePosition(self, T, num):
        """Calculate the sinusoidal positional encoding based on the relative position of each considered frame.
        r_pos calculations as here: https://theaisummer.com/positional-embeddings/

        :param int T: Number of frames contained in Q, K and V
        :return: Tensor with shape [T, T]
        """
        freq = self.freq
        d = 2 * T
        min_rpos = -(T - 1)

        m = num

        i = []
        j = []

        #i = torch.tensor([k for k in range(T)], device=self.out.weight.device)
        #j = torch.tensor([k for k in range(T)], device=self.out.weight.device)
        for _ in range(m):
            i_tmp = torch.tensor([k for k in range(T)], device=self.out.weight.device)
            j_tmp = torch.tensor([k for k in range(T)], device=self.out.weight.device)
            i.append(i_tmp)
            j.append(j_tmp)

        i = torch.stack(i, dim=0)
        j = torch.stack(j, dim=0)

        # Reshape tensors each i for each j indices
        i = i.reshape(m, i.shape[1], 1)
        i = i.repeat_interleave(i.shape[1], dim=2)
        j = j.repeat(m, i.shape[1], 1)

        # Calculate the relative positions
        r_pos = j - i - min_rpos

        RP = torch.zeros(m, T, T, device=self.out.weight.device)
        idx = torch.tensor([k for k in range(T//2)], device=self.out.weight.device)
        #RP[:, 2*idx] = torch.sin(r_pos[:, 2*idx] / freq ** ((i[:, 2*idx] + j[:, 2*idx]) / d))
        #RP[:, 2*idx+1] = torch.cos(r_pos[:, 2*idx+1] / freq ** ((i[:, 2*idx+1] + j[:, 2*idx+1]) / d))
        for k in range(m):
            RP[k, :, 2 * idx] = torch.sin(r_pos[k, :, 2 * idx] / freq ** ((i[k, :, 2 * idx] + j[k, :, 2 * idx]) / d))
            RP[k, :, 2*idx+1] = torch.cos(r_pos[k, :, 2*idx+1] / freq ** ((i[k, :, 2*idx+1] + j[k, :, 2*idx+1]) / d))
        return RP

    @staticmethod
    def get_entropy(logits):
        """ Compute the entropy for each row of the attention matrix.

        :param torch.Tensor logits: The raw (non-normalized) attention values with shape [sacles*scalses, T, T].
        :return: A torch.Tensor containing the normalized entropy of each row of the attention matrix, with shape [sacles*scalses, T].
        """
        # F.softmax():按照行或者列来做归一化的。https://www.zhihu.com/question/456213566
        # F.log_softmax():在softmax的结果上再做多一次log运算。https://blog.csdn.net/m0_51004308/article/details/118001835
        '''https://blog.51cto.com/u_15075507/4403348.
           https://liubingqing.blog.csdn.net/article/details/118421352?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-6-118421352-blog-118001835.235%5Ev27%5Epc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-6-118421352-blog-118001835.235%5Ev27%5Epc_relevant_multi_platform_whitelistv3&utm_relevant_index=9
           此处用的方法：用log_softmax()+nll_loss()实现
        '''
        _entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        _entropy = -1.0 * _entropy.sum(-1)

        # https://stats.stackexchange.com/a/207093 Maximum value of entropy is log(k), where k the # of used categories.
        # Here k is when all the values of a row is different of each other (i.e., k = # of video frames)
        return _entropy / np.log(logits.shape[1])

    def forward(self, x):
        """ Compute the weighted frame features, based on either the global or local (multi-head) attention mechanism.

        :param torch.tensor x: Frame features with shape [T, input_size]
        :return: A tuple of:
                    y: Weighted features based on the attention weights, with shape [T, input_size]
                    att_weights : The attention weights (before dropout), with shape [T, T]
        """

        # input shape: [seq_len, input_size]
        h, w = x.size(0), x.size(1) // self.heads

        local_y = []
        local_x = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale - 1):
                    end_x = h
                if j == (self.scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        #  self-attention func
        def func(value_local, query_local, key_local, diversity):
            # size(0)返回x的维度形状
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(1), value_local.size(2)
            #value_local = value_local.contiguous().view(batch_size_new, self.value_channels, -1)

            #query_local = query_local.contiguous().view(batch_size_new, self.key_channels, -1)
            #query_local = query_local.permute(0, 2, 1)
            #key_local = key_local.contiguous().view(batch_size_new, self.key_channels, -1)
            key_local = key_local.permute(0, 2, 1)

            sim_map = torch.bmm(query_local, key_local)  # batch matrix multiplication
            #sim_map = (self.key_channels ** -.5) * sim_map
            #sim_map = F.softmax(sim_map, dim=-1)
            if self.pos_enc is not None:
                if self.pos_enc == "absolute":
                    AP = self.getAbsolutePosition(T=sim_map.shape[1], num=batch_size_new)
                    sim_map = sim_map + AP
                elif self.pos_enc == "relative":
                    RP = self.getRelativePosition(T=sim_map.shape[1], num=batch_size_new)
                    sim_map = sim_map + RP
            att_weights = self.softmax(sim_map)

            #entropy: shape: [scales*scales, T]
            entropy = self.get_entropy(logits=sim_map)
            entropy = F.normalize(entropy, p=1, dim=-1)

            attn_remainder = att_weights
            div_remainder = diversity
            dep_factor = (div_remainder * attn_remainder).sum(-1).div(div_remainder.sum(-1))
            dep_factor = dep_factor.unsqueeze(1).expand(-1, dep_factor.shape[1], -1)
            masked_dep_factor = dep_factor * att_weights
            att_win = att_weights + masked_dep_factor
            #sim_map = self.drop(sim_map)

            #context_local = torch.bmm(value_local, sim_map.permute(0, 2, 1))
            #context_local: [scale*scale, T/sacle, input/heads/scale]
            context_local = torch.bmm(att_win, value_local)
            # context_local = context_local.permute(0, 2, 1).contiguous()
            #context_local = context_local.view(batch_size_new, h_local, w_local)
            #characteristics = (entropy, dep_factor[0, :])
            #characteristics = torch.stack(characteristics).detach()
            characteristics = (entropy[:, :], dep_factor[:, 0, :])
            characteristics = torch.stack(characteristics, dim=1).detach()
            outputs_y = torch.cat(tensors=(context_local, characteristics.permute(0, 2, 1)), dim=-1)
            y_s = self.out_2(outputs_y)

            #return context_local
            return y_s

        x_unit = F.normalize(x, p=2, dim=1)
        # .t():将Tensor进行转置。x @ x:是torch.matmul（）的重写
        similarity = x_unit @ x_unit.t()
        diversity = 1 - similarity

        outputs = []
        for head in range(self.heads):
            K = self.Wk[head](x)
            Q = self.Wq[head](x)
            V = self.Wv[head](x)

            local_block_cnt_2 = 2 * self.scale * self.scale
            local_block_cnt = local_block_cnt_2 - (2*self.scale)
            #  Parallel Computing to speed up
            #  reshape value_local, q, k
            v_list = [V[local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in
                      range(0, local_block_cnt, 2)]
            v_locals = torch.stack(v_list, dim=0)
            q_list = [Q[local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in
                      range(0, local_block_cnt, 2)]
            q_locals = torch.stack(q_list, dim=0)
            k_list = [K[local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in
                      range(0, local_block_cnt, 2)]
            k_locals = torch.stack(k_list, dim=0)
            d_list = [diversity[local_x[i]:local_x[i + 1], local_x[i]:local_x[i + 1]] for i in
                      range(0, local_block_cnt, 2)]
            d_locals = torch.stack(d_list, dim=0)
            #v_list, q_list, k_list: [scale*scale, T/scale, input/heads/scale]
            # print(v_locals.shape)
            context_locals = func(v_locals, q_locals, k_locals, d_locals)

            v_list_2 = [V[local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in
                      range(local_block_cnt, local_block_cnt_2, 2)]
            v_locals_2 = torch.stack(v_list_2, dim=0)
            q_list_2 = [Q[local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in
                      range(local_block_cnt, local_block_cnt_2, 2)]
            q_locals_2 = torch.stack(q_list_2, dim=0)
            k_list_2 = [K[local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in
                      range(local_block_cnt, local_block_cnt_2, 2)]
            k_locals_2 = torch.stack(k_list_2, dim=0)
            d_list_2 = [diversity[local_x[i]:local_x[i + 1], local_x[i]:local_x[i + 1]] for i in
                        range(local_block_cnt, local_block_cnt_2, 2)]
            d_locals_2 = torch.stack(d_list_2, dim=0)

            context_locals_2 = func(v_locals_2, q_locals_2, k_locals_2, d_locals_2)

            context_list = []
            for i in range(0, self.scale-1):
                row_tmp = []
                for j in range(0, self.scale):
                    left = 1 * (j + i * self.scale)
                    right = 1 * (j + i * self.scale) + 1
                    tmp = context_locals[left:right]
                    row_tmp.append(tmp)
                context_list.append(torch.cat(row_tmp, 2))

            #context: [1, T, input/heads]
            context_1 = torch.cat(context_list, 1)

            context_list_2 = []
            for i in range(1):
                row_tmp = []
                for j in range(0, self.scale):
                    left = 1 * (j + i * self.scale)
                    right = 1 * (j + i * self.scale) + 1
                    tmp = context_locals_2[left:right]
                    row_tmp.append(tmp)
                context_list_2.append(torch.cat(row_tmp, 2))

            # context: [1, T, input/heads]
            context_2 = torch.cat(context_list_2, 1)
            #context: [1, T, input/heads]
            context = torch.cat((context_1, context_2), 1)
            #y: [T, input/heads]
            y_heads = context.squeeze(0)

            # Save the current head output
            outputs.append(y_heads)
        y = self.out(torch.cat(outputs, dim=1))
        return y


            # Q *= 0.06                       # scale factor VASNet
            # Q /= np.sqrt(self.output_size)  # scale factor (i.e 1 / sqrt(d_k) )
        #     energies = torch.matmul(Q, K.transpose(1, 0))
        #     if self.pos_enc is not None:
        #         if self.pos_enc == "absolute":
        #             AP = self.getAbsolutePosition(T=energies.shape[0])
        #             energies = energies + AP
        #         elif self.pos_enc == "relative":
        #             RP = self.getRelativePosition(T=energies.shape[0])
        #             energies = energies + RP
        #
        #     att_weights = self.softmax(energies)
        #     _att_weights = self.drop(att_weights)
        #     y = torch.matmul(_att_weights, V)
        #
        #     # Save the current head output
        #     outputs.append(y)
        # y = self.out(torch.cat(outputs, dim=1))
        # return y, att_weights.clone()  # for now we don't deal with the weights (probably max or avg pooling)


if __name__ == '__main__':
    pass
    """Uncomment for a quick proof of concept
    model = SelfAttention(input_size=256, output_size=256, pos_enc="absolute").cuda()
    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")
    """
