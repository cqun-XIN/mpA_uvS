import torch
import torch.nn.functional as F
import torch.nn as nn


entropy = torch.ones(3, 5)
print(entropy, entropy.shape)
entropy = F.normalize(entropy, p=1, dim=-1)
print(entropy, entropy.shape)

# attn_remainder_ = torch.ones(5, 5)
# div_remainder_ = torch.ones(5, 5)
# att_weights = torch.ones(5 ,5)
# print(div_remainder_, div_remainder_.shape)
# print(attn_remainder_, attn_remainder_.shape)
#
#
#
#
#
# dep_factor_ = (div_remainder_ * attn_remainder_).sum(-1).div(div_remainder_.sum(-1))
# print(dep_factor_, dep_factor_.shape)
# dep_factor_ = dep_factor_.unsqueeze(0).expand(dep_factor_.shape[0], -1)
# #dep_factor_ = dep_factor_.unsqueeze(0)
# print(dep_factor_, dep_factor_.shape)
#
# att_win = att_weights + dep_factor_
# print(att_win, att_win.shape)
# characteristics = (entropy, dep_factor_[0, :])
# characteristics = torch.stack(characteristics).detach()
# print(characteristics, characteristics.shape)

attn_remainder = torch.ones(3, 5, 5)
div_remainder = torch.ones(3, 5, 5)
att_weights = torch.ones(3, 5 ,5)

dep_factor = (div_remainder * attn_remainder).sum(-1).div(div_remainder.sum(-1))
print(dep_factor, dep_factor.shape)
dep_factor = dep_factor.unsqueeze(1).expand(-1, dep_factor.shape[1], -1)
#dep_factor = dep_factor.unsqueeze(1)
print(dep_factor, dep_factor.shape)

att_win = att_weights + dep_factor
print(att_win, att_win.shape)
characteristics = (entropy[:,:], dep_factor[:, 0, :])
characteristics = torch.stack(characteristics, dim=1).detach()
print(characteristics, characteristics.shape)
print(characteristics.permute(0, 2, 1), characteristics.permute(0, 2, 1).shape)

context_local =torch.ones(3, 5, 10)
x = torch.cat(tensors=(context_local, characteristics.permute(0, 2, 1)), dim=-1)
print(x, x.shape)


out_2 = nn.Linear(in_features=12, out_features=10, bias=False)
print(out_2)
y = out_2(x)
print(x, x.shape)
# y = torch.ones(5, 10)
# outputs = torch.cat(tensors=(y, characteristics.t()), dim=-1)
# print(outputs, outputs.shape)

#out_2 = nn.Linear(in_features=(1024//4)+2, out_features=1024//4, bias=False)
out = nn.Linear(in_features=1024//4//4+2, out_features=1024//4//4, bias=False)
print(out)

# local_y = []
# local_x = []
# step_h, step_w = 100 // 4, 1024//4 // 4
# for i in range(0, 4):
#     for j in range(0, 4):
#         start_x, start_y = i * step_h, j * step_w
#         end_x, end_y = min(start_x + step_h, 100), min(start_y + step_w, 1024//4)
#         if i == (4 - 1):
#             end_x = 100
#         if j == (4 - 1):
#             end_y = 1024//4
#         local_x += [start_x, end_x]
#         local_y += [start_y, end_y]
# print(local_y)
# print(local_x)
#
# value=torch.ones(1,8,100,1024)
# local_block_cnt=2*4*4
# v_list = [value[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
# print(len(v_list))
# v_locals = torch.cat(v_list,dim=0)
# print(v_locals.shape)
#
# value=torch.ones(100,1024)
# local_block_cnt=2*4*4
# v_list = [value[local_x[i]:local_x[i+1],local_x[i]:local_x[i+1]] for i in range(0, local_block_cnt, 2)]
# print(len(v_list))
# v_locals = torch.stack(v_list,dim=0)
# print(v_locals.shape)

# entropy = torch.ones(5)
# print(entropy, entropy.shape)
# entropy = F.normalize(entropy, p=1, dim=-1)
# print(entropy, entropy.shape)

# logits = torch.ones(5 ,5)
# print(logits)
# _entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
# print(_entropy,_entropy.shape)
# _entropy = -1.0 * _entropy.sum(-1)
# print(_entropy,_entropy.shape)

# entropy = torch.ones(3, 5, 5)
# print(entropy, entropy.shape)
# entropy = F.normalize(entropy, p=2, dim=2)
# print(entropy, entropy.shape)


# value_local = v_locals
# batch_size_new = value_local.size(0)
# h_local, w_local = value_local.size(2), value_local.size(3)
# value_local = value_local.contiguous().view(batch_size_new, 256, -1)
# print(value_local.shape)

# scale = 4
# context_locals = torch.ones(12, 25, 256)
# print(context_locals.shape)
# context_list = []
# for i in range(0, scale-1):
#     row_tmp = []
#     for j in range(0, scale):
#         left = 1 * (j + i * scale)
#         right = 1 * (j + i * scale) + 1
#         tmp = context_locals[left:right]
#         row_tmp.append(tmp)
#     context_list.append(torch.cat(row_tmp, 2))
#
# context = torch.cat(context_list, 1)
# print(context.shape)
#
# context_locals_2 = torch.ones(4, 26, 256)
# context_list_2 = []
# for i in range(1):
#     row_tmp = []
#     for j in range(0, scale):
#         left = 1 * (j + i * scale)
#         right = 1 * (j + i * scale) + 1
#         tmp = context_locals_2[left:right]
#         row_tmp.append(tmp)
#     context_list_2.append(torch.cat(row_tmp, 2))
#
#  # context: [1, T, input/heads]
# context_2 = torch.cat(context_list_2, 1)
# print(context_2.shape)
# context_1 = torch.cat((context, context_2), 1)
# print(context_1.shape)
'''
m = 3
energies = torch.ones(5, 10)
T=energies.shape[0]
print(T)
pos = []
# i = []
for _ in range(m):
    pos_tmp = torch.tensor([k for k in range(T)])
    #i_tmp = torch.tensor([k for k in range(T // 2)])
    pos.append(pos_tmp)
    #i.append(i_tmp)
pos = torch.stack(pos,dim=0)
#i = torch.stack(i, dim=0)
i = torch.tensor([k for k in range(T // 2)])
# pos = torch.tensor([k for k in range(T)])
# i = torch.tensor([k for k in range(T // 2)])

print(pos,pos.shape)
print(i,i.shape)
pos = pos.reshape(m,pos.shape[1], 1)
print(pos,pos.shape)
pos = pos.repeat_interleave(i.shape[0], dim=2)
print(pos,pos.shape)
i = i.repeat(m, pos.shape[1], 1)
print(i,i.shape)
AP = torch.zeros(m, T, T)
print(AP,AP.shape)
# print(2*i)
# print(pos/10000, (pos/10000).shape)
for j in range(m):
    AP[j, pos, 2 * i] = pos / 10000 ** ((2 * i) / 1024)
print(AP)
'''