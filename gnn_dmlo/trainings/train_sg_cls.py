import os, wandb, glob
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from gnn_dmlo.model.network import SubGraphClassificator
from gnn_dmlo.core.utils import set_seeds, count_parameters, print_log, update_log
from gnn_dmlo.trainings.dataset_sg_cls import GraphDatasetNodeCls
import numpy as np

##########################################

config = dict(
    max_iters=50000,
    batch_size=64,
    lr=0.00001,
    hidden_dim=128,
    enc_nodes_dim=32,
    mask_enc=True,
    lp_fea_enc=True,
    edge_dir_enc=True,
    edge_norm_enc=False,
    gnn_layers=1,
    dataset_train_path="path_to_train",
    dataset_val_path="path_to_val",
    seed=0,
)

wandb.init(config=config, project="xxx", entity="xxx", mode="online")
config = wandb.config

##########################################


def dice_score(a, b):
    intersection = np.logical_and(a, b)
    s = 2.0 * intersection.sum() / (a.sum() + b.sum())
    if np.isnan(s):
        s = 0.0
    return s


class Trainer:
    def __init__(self, config):
        self.proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device ", self.device)

        print("*" * 20)
        for k, v in config.items():
            print(f"\t{k}: {v}")
        print("*" * 20)

        set_seeds(config["seed"])

        self.global_step = 0

        self.ckpt_name = f"nodecls_{wandb.run.name}.pth"

        #####################
        # DATASET
        dataset_train = GraphDatasetNodeCls(os.path.join(self.proj_path, "datasets", config["dataset_train_path"]))
        self.train_loader = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True, drop_last=True)

        dataset_val = GraphDatasetNodeCls(os.path.join(self.proj_path, "datasets", config["dataset_val_path"]))
        self.val_loader = DataLoader(dataset_val, batch_size=config["batch_size"], shuffle=False, drop_last=True)

        print("train: ", len(self.train_loader) * config["batch_size"])
        print("val: ", len(self.val_loader) * config["batch_size"])

        #####################

        # MODEL
        self.model = SubGraphClassificator(config)
        self.model = self.model.to(self.device)
        print("MODEL PARAMETERS: ", count_parameters(self.model))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train_loop(self):
        # ************************
        # TRAINING
        min_val_loss = torch.inf

        for epoch in range(1, 1000):
            log_dict = {
                "epoch": epoch,
                "train_loss_cls": 0.0,
                "val_loss_cls": 0.0,
                "test_loss_cls": 0.0,
                "val_dice": 0.0,
                "test_dice": 0.0,
            }

            # train
            self.model.train()
            for data in tqdm(self.train_loader):
                data = data.to(self.device)
                t_dict = self.train(data)
                update_log(t_dict, log_dict)

            # val
            self.model.eval()
            for data in self.val_loader:
                data = data.to(self.device)
                v_dict = self.val(data)
                update_log(v_dict, log_dict)

            # compute averages
            for k, v in log_dict.items():
                if "epoch" in k:
                    continue
                elif "train" in k:
                    log_dict[k] /= len(self.train_loader)
                elif "val" in k:
                    log_dict[k] /= len(self.val_loader)
                elif "test" in k:
                    log_dict[k] /= len(self.test_loader)

            # logging
            print_log(log_dict)
            print("step: ", self.global_step)
            wandb.log(log_dict, step=self.global_step)

            # best model
            if log_dict["val_loss_cls"] < min_val_loss:
                min_val_loss = log_dict["val_loss_cls"]
                print("* Best Model!*")

                # saving
                state = dict(self.config)
                state["model"] = self.model.state_dict()
                torch.save(state, os.path.join(self.proj_path, "checkpoints", self.ckpt_name))

            if self.config["max_iters"] < self.global_step:
                print("**max iters reached!**")
                break

    def train(self, batch):
        self.optimizer.zero_grad()

        z_cls = self.model(batch).squeeze()
        loss = self.criterion(z_cls, batch.label.float())

        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return {"train_loss_cls": loss}

    @torch.no_grad()
    def val(self, batch):
        z_cls = self.model(batch).squeeze()
        loss = self.criterion(z_cls, batch.label.float())

        pred_cls = torch.sigmoid(z_cls).detach().cpu().numpy().squeeze()
        gt_cls = batch.label.squeeze().detach().cpu().numpy()
        val_dice = dice_score(np.where(pred_cls > 0.5, 1, 0), gt_cls)

        return {"val_loss_cls": loss, "val_dice": val_dice}


if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train_loop()
