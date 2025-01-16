# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, freq=10000, heads=1, pos_enc=None):
        """ The basic (multi-head) Attention 'cell' containing the learnable parameters of Q, K and V

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param int heads: Number of heads for the attention module.
        :param str | None pos_enc: The type of the positional encoding [supported: Absolute, Relative].
        """
        super(SelfAttention, self).__init__()

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
        '''nn.ModuleList()类似于n.Sequential()，但不相同。https://zhuanlan.zhihu.com/p/75206669'''
        self.Wk, self.Wq, self.Wv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(self.heads):
            '''output_size//heads：表示向下取整的除法。https://blog.csdn.net/SummerMiko_/article/details/123428734'''
            self.Wk.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
            self.Wq.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
            self.Wv.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
        self.out = nn.Linear(in_features=output_size, out_features=input_size, bias=False)
        self.out_2 = nn.Linear(in_features=(output_size//heads)+2, out_features=output_size//heads, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=0.5)

    def getAbsolutePosition(self, T):
        """Calculate the sinusoidal positional encoding based on the absolute position of each considered frame.
        Based on 'Attention is all you need' paper (https://arxiv.org/abs/1706.03762)

        :param int T: Number of frames contained in Q, K and V
        :return: Tensor with shape [T, T]
        """
        freq = self.freq
        d = self.input_size

        pos = torch.tensor([k for k in range(T)], device=self.out.weight.device)
        i = torch.tensor([k for k in range(T//2)], device=self.out.weight.device)

        # Reshape tensors each pos_k for each i indices
        pos = pos.reshape(pos.shape[0], 1)
        '''
           repeat_interleave是将张量中的元素沿某一维度复制n次，即复制后的张量沿该维度相邻的n个元素是相同的。
           repeat_interleave()中有两个参数，第一个参数N是代表复制多少次，第二个参数代表维度。
           https://blog.csdn.net/starlet_kiss/article/details/125718922
        '''
        pos = pos.repeat_interleave(i.shape[0], dim=1)
        '''复制的一种方式，与repeat_interleave略有不同。https://blog.csdn.net/starlet_kiss/article/details/125718922'''
        i = i.repeat(pos.shape[0], 1)

        AP = torch.zeros(T, T, device=self.out.weight.device)
        AP[pos, 2*i] = torch.sin(pos / freq ** ((2 * i) / d))
        AP[pos, 2*i+1] = torch.cos(pos / freq ** ((2 * i) / d))
        return AP

    def getRelativePosition(self, T):
        """Calculate the sinusoidal positional encoding based on the relative position of each considered frame.
        r_pos calculations as here: https://theaisummer.com/positional-embeddings/

        :param int T: Number of frames contained in Q, K and V
        :return: Tensor with shape [T, T]
        """
        freq = self.freq
        d = 2 * T
        min_rpos = -(T - 1)

        i = torch.tensor([k for k in range(T)], device=self.out.weight.device)
        j = torch.tensor([k for k in range(T)], device=self.out.weight.device)

        # Reshape tensors each i for each j indices
        i = i.reshape(i.shape[0], 1)
        i = i.repeat_interleave(i.shape[0], dim=1)
        j = j.repeat(i.shape[0], 1)

        # Calculate the relative positions
        r_pos = j - i - min_rpos

        RP = torch.zeros(T, T, device=self.out.weight.device)
        idx = torch.tensor([k for k in range(T//2)], device=self.out.weight.device)
        RP[:, 2*idx] = torch.sin(r_pos[:, 2*idx] / freq ** ((i[:, 2*idx] + j[:, 2*idx]) / d))
        RP[:, 2*idx+1] = torch.cos(r_pos[:, 2*idx+1] / freq ** ((i[:, 2*idx+1] + j[:, 2*idx+1]) / d))
        return RP

    @staticmethod
    def get_entropy(logits):
        """ Compute the entropy for each row of the attention matrix.

        :param torch.Tensor logits: The raw (non-normalized) attention values with shape [T, T].
        :return: A torch.Tensor containing the normalized entropy of each row of the attention matrix, with shape [T].
        """
        #F.softmax():按照行或者列来做归一化的。https://www.zhihu.com/question/456213566
        #F.log_softmax():在softmax的结果上再做多一次log运算。https://blog.csdn.net/m0_51004308/article/details/118001835
        '''https://blog.51cto.com/u_15075507/4403348.
           https://liubingqing.blog.csdn.net/article/details/118421352?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-6-118421352-blog-118001835.235%5Ev27%5Epc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-6-118421352-blog-118001835.235%5Ev27%5Epc_relevant_multi_platform_whitelistv3&utm_relevant_index=9
           此处用的方法：用log_softmax()+nll_loss()实现
        '''
        _entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        _entropy = -1.0 * _entropy.sum(-1)

        # https://stats.stackexchange.com/a/207093 Maximum value of entropy is log(k), where k the # of used categories.
        # Here k is when all the values of a row is different of each other (i.e., k = # of video frames)
        return _entropy / np.log(logits.shape[0])

    def forward(self, x):
        """ Compute the weighted frame features, based on either the global or local (multi-head) attention mechanism.

        :param torch.tensor x: Frame features with shape [T, input_size]
        :return: A tuple of:
                    y: Weighted features based on the attention weights, with shape [T, input_size]
                    att_weights : The attention weights (before dropout), with shape [T, T]
        """

        # Compute the pairwise dissimilarity of each frame, on the initial feature space (GoogleNet features)
        # 计算每个帧在初始特征空间上的成对相异性（GoogleNet特征）
        '''L2 norm计算:https://blog.csdn.net/lj2048/article/details/118115681'''
        x_unit = F.normalize(x, p=2, dim=1)
        # .t():将Tensor进行转置。x @ x:是torch.matmul（）的重写
        similarity = x_unit @ x_unit.t()
        diversity = 1 - similarity

        outputs = []
        for head in range(self.heads):
            K = self.Wk[head](x)
            Q = self.Wq[head](x)
            V = self.Wv[head](x)

            # Q *= 0.06                       # scale factor VASNet
            # Q /= np.sqrt(self.output_size)  # scale factor (i.e 1 / sqrt(d_k) )
            energies = torch.matmul(Q, K.transpose(1, 0))
            if self.pos_enc is not None:
                if self.pos_enc == "absolute":
                    AP = self.getAbsolutePosition(T=energies.shape[0])
                    energies = energies + AP
                elif self.pos_enc == "relative":
                    RP = self.getRelativePosition(T=energies.shape[0])
                    energies = energies + RP

            att_weights = self.softmax(energies)

            # Entropy is a measure of uncertainty: Higher value means less information.熵是一种不确定性的衡量标准：更高的值意味着更少的信息。
            entropy = self.get_entropy(logits=energies)
            entropy = F.normalize(entropy, p=1, dim=-1)

            attn_remainder = att_weights
            div_remainder = diversity
            dep_factor = (div_remainder * attn_remainder).sum(-1).div(div_remainder.sum(-1))
            dep_factor = dep_factor.unsqueeze(0).expand(dep_factor.shape[0], -1)
            masked_dep_factor = dep_factor * att_weights
            att_win = att_weights + masked_dep_factor

            _att_weights = self.drop(att_win)
            #_att_weights = self.drop(att_weights)
            y = torch.matmul(_att_weights, V)

            characteristics = (entropy, dep_factor[0, :])
            characteristics = torch.stack(characteristics).detach()
            outputs_y = torch.cat(tensors=(y, characteristics.t()), dim=-1)

            y_s = self.out_2(outputs_y)
            # Save the current head output
            outputs.append(y_s)
        y = self.out(torch.cat(outputs, dim=1))
        return y, att_win.clone()
        #return y, att_weights.clone()  # for now we don't deal with the weights (probably max or avg pooling)


if __name__ == '__main__':
    pass
    """Uncomment for a quick proof of concept
    model = SelfAttention(input_size=256, output_size=256, pos_enc="absolute").cuda()
    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")
    """
