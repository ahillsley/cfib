#%%
import torch
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import DataLoader
from dataloader import scDataSet
from tqdm import tqdm

#%%
resnet = models.resnet50(pretrained=True)

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
#%%

ds = scDataSet('/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/data/analysis/2-dataset/pilot_1_dataset.zarr')
loader = DataLoader(
    ds,
    batch_size=30,
    shuffle=True
)
# %%
for epoch in range(5):
    for item, _ in tqdm(loader):
        loss = learner(item.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()
torch.save(resnet.state_dict(), './improved-net.pt')
# %%
resnet = models.resnet50()
resnet.load_state_dict(torch.load('./improved-net.pt'))
embedding_list = []
label_list = []
for item, label in tqdm(loader):
    projection, embeddings = learner(item.float(), return_embedding = True)
    embedding_list.append(embeddings)
    label_list.append(label)
    if len(embedding_list) == 30:
        break
# %%
import umap
reducer = umap.UMAP()
features = torch.concatenate(embedding_list)
labels = [item for tup in label_list for item in tup]
labels = [int(label_dict[z][0]) for z in labels] # from visualize_results
a = [l == 2 for l in labels]
reduced_features = reducer.fit_transform(features.detach().numpy())

import matplotlib.pyplot as plt
plt.scatter(
    reduced_features[:,0],
    reduced_features[:,1],
    c = labels
)
# %%
