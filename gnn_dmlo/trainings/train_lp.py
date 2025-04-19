import os, wandb, torch
from tqdm import tqdm
import numpy as np
from torch_geometric.loader import DataLoader
from gnn_dmlo.model.network import Network
from gnn_dmlo.core.utils import set_seeds, count_parameters, print_log, update_log
from gnn_dmlo.trainings.dataset_lp import GraphDatasetLinkPred


##########################################

CONFIG = dict(
    batch_size=2,
    epochs=5000,
    max_iters=300000,
    lr=0.0001,
    hidden_dim=32,
    edge_norm_enc=False,
    edge_dir_enc=True,
    activation="relu",
    dataset_train_path="path_to_train",
    dataset_val_path="path_to_val",
    seed=0,
)

wandb.init(config=CONFIG, project="xxx", entity="xxx", mode="online")
config = wandb.config

##########################################


def dice_score(a, b):
    intersection = np.logical_and(a, b)
    return 2.0 * intersection.sum() / (a.sum() + b.sum())


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

        #####################
        # DATASET
        train_path = os.path.join(self.proj_path, "datasets", self.config["dataset_train_path"])
        val_path = os.path.join(self.proj_path, "datasets", self.config["dataset_val_path"])
        self.train_loader = DataLoader(
            GraphDatasetLinkPred(train_path), batch_size=self.config["batch_size"], shuffle=True
        )
        self.val_loader = DataLoader(GraphDatasetLinkPred(val_path), batch_size=1, shuffle=False)

        print("train: ", len(self.train_loader) * self.config["batch_size"])
        print("val: ", len(self.val_loader))

        # MODEL
        self.model = Network(self.config)
        self.model = self.model.to(self.device)
        print("MODEL PARAMETERS: ", count_parameters(self.model))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def train_loop(self):
        # ************************
        # TRAINING
        min_val_loss = torch.inf
        for epoch in range(1, self.config["epochs"] + 1):
            log_dict = {
                "epoch": epoch,
                "train_tot_loss": 0.0,
                "train_loss_lp": 0.0,
                "train_loss_angle": 0.0,
                "val_tot_loss": 0.0,
                "val_loss_lp": 0.0,
                "val_loss_angle": 0.0,
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
            for k, _ in log_dict.items():
                if "epoch" in k:
                    continue
                elif "train" in k:
                    log_dict[k] /= len(self.train_loader)
                else:
                    log_dict[k] /= len(self.val_loader)

            #############################################################

            # logging
            print_log(log_dict)
            print("step: ", self.global_step)
            wandb.log(log_dict, step=self.global_step)

            # best model
            if log_dict["val_tot_loss"] < min_val_loss:
                min_val_loss = log_dict["val_tot_loss"]
                print("* Best Model!*")

                # saving
                state = dict(self.config)
                state["model"] = self.model.state_dict()
                torch.save(state, os.path.join(self.proj_path, "checkpoints", f"linkpred_{wandb.run.name}.pth"))

            if self.config["max_iters"] < self.global_step:
                print("**max iters reached!**")
                break

    def train(self, batch):
        self.optimizer.zero_grad()

        out = self.model(batch)

        loss_lp, loss_angle = self.model.losses(batch, out["node_feat"], out["node_angle"])
        loss = loss_lp + loss_angle

        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return {"train_tot_loss": loss, "train_loss_lp": loss_lp, "train_loss_angle": loss_angle}

    @torch.no_grad()
    def val(self, batch):
        out = self.model(batch)

        loss_lp, loss_angle = self.model.losses(batch, out["node_feat"], out["node_angle"])
        loss = loss_lp + loss_angle

        return {"val_tot_loss": loss, "val_loss_lp": loss_lp, "val_loss_angle": loss_angle}


if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train_loop()
