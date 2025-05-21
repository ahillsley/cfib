import torch
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from dataloader import scDataSet
from tqdm import tqdm
import wandb


def train(
        dataset: Dataset,
        config: dict,

)-> None:
    resnet = models.resnet50(weights='DEFAULT')
    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool'
    )
    opt = torch.optim.Adam(learner.parameters(), lr=config["lr"])

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    for epoch in range(config['epochs']):
        for item, _ in tqdm(loader):
            loss = learner(item.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
            wandb.log({"loss": loss})

        torch.save(learner.online_encoder.state_dict(), f'{config["model_save_path"]}/improved-net_{epoch}.pt')

    return

if __name__ == "__main__":
    config = {
        "batch_size": 30,
        "epochs": 11,
        "itters": 5_000,
        "model_save_path": '/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/cfib/model/models',
        "lr": 3e-4,
        "dataset_path": '/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/data/analysis/2-dataset/pilot_1_dataset.zarr'
    }

    run = wandb.init(
        project = "cfib",
        config={
            "learning_rate": config['lr'],
            "epochs": config['epochs'],
        }
    )

    ds = scDataSet(config["dataset_path"])

    train(
        dataset=ds,
        config=config
    )
