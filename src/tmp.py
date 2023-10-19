import torch

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def euclidean_distance(i, vec, w):
    return torch.sqrt(torch.square(vec//w-i//w)+torch.square(vec%w-i%w))

x_embed_1 = torch.rand(2,3,2,2)
x_embed_2 = torch.rand(2,3,2,2)
x_assembly = torch.rand(2,3,2,2)
k = 2

N, C, H, W = x_embed_1.shape
x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C))
x_embed_2 = x_embed_2.view(N, C, H * W)

score = torch.matmul(x_embed_1, x_embed_2)
score = torch.softmax(score, -1)  # N, H*W, H*W

# ind_vec = torch.arange(H * W).unsqueeze(1).repeat(1,H*W).unsqueeze(0).repeat(2,1,1)
# ind_vec1 = torch.cat([torch.arange(H * W).unsqueeze(1).repeat(1,H*W).unsqueeze(0), torch.arange(H * W).unsqueeze(0).repeat(H*W,1).unsqueeze(0)],dim=0)
# print(ind_vec)
# print(ind_vec1)
# distance_map = torch.mul(ind_vec1-ind_vec, ind_vec1-ind_vec)
# distance_map = torch.sqrt(distance_map[0,:,:] + distance_map[1,:,:]).unsqueeze(0).repeat(2,1,1)
# print(distance_map)
distance_map = torch.zeros_like(score)
ind_vec = torch.arange(H * W)
for i in range(H*W):
    distance_map[:, i, :] = euclidean_distance(i, ind_vec, W)
distance_map = distance_map / torch.max(distance_map, dim=1)[0].unsqueeze(2)

d_query = score * distance_map
d_query = torch.sum(d_query, dim=2, keepdims=True) # N, H*W, 1
_, d_query_topk_ind = torch.topk(d_query, k=k, dim=1)

selected_score = batched_index_select(score, d_query_topk_ind.squeeze())

x_assembly = x_assembly.view(N, -1, H * W).permute(0, 2, 1)
x_final = torch.matmul(selected_score, x_assembly)
x_embed_1[torch.arange(N).unsqueeze(1), d_query_topk_ind[:, :, 0], :] = x_final

# a = torch.LongTensor([[[2, 1], [2, 2], [3, 3]], [[2, 0], [3, 2], [1, 2]]])
# K = 5
# out = torch.zeros(a.size(0), K, K)
# print(torch.arange(out.size(0)).unsqueeze(1).size())
# out[torch.arange(out.size(0)).unsqueeze(1), a[:, :, 0], a[:, :, 1]] = 1
# print(out)

# recovered_query = batched_index_select(selected_query, undo_d_query_topk_ind.squeeze())
