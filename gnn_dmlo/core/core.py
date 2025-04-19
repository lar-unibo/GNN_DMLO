import numpy as np
import os
import torch, torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

from gnn_dmlo.model.network import Network, SubGraphClassificator
from gnn_dmlo.core.graph_generation import GraphGenerationWh
from gnn_dmlo.core.utils import get_data_pyg, set_seeds, to_numpy, compute_data_subgraph
from gnn_dmlo.core.solver import SolverLPCheck, SolverSgCls
from gnn_dmlo.core.paths_extractor import PathsProc
from gnn_dmlo.core.paths_aggregator import PathsAggregator


def ellipse(center, cov):
    u = center[0]  # x-position of the center
    v = center[1]  # y-position of the center

    a = np.sqrt(cov[0, 0])  # radius on the x-axis
    b = np.sqrt(cov[1, 1])  # radius on the y-axis

    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(u + a * np.cos(t), v + b * np.sin(t))


class FullPredictor:
    def __init__(self, checkpoint_link_pred, checkpoint_sg_cls):
        self.link_angle_predictor = LinkAnglePredictor(checkpoint_link_pred=checkpoint_link_pred)
        self.sg_cls_predictor = SgClsPredictor(checkpoint_cls=checkpoint_sg_cls)
        self.path_ext = PathsProc()
        self.aggregator = PathsAggregator()

    def run(self, mask_img, plot=False):
        # link and angle pred
        data_pyg, out_dict = self.link_angle_predictor.run(mask_img)

        # subgraph cls pred
        graph, node_cls, clusters = self.sg_cls_predictor.run(data_pyg, out_dict)

        # paths

        paths, graphs = self.path_ext.run(graph, mask_img)

        paths_final, out_int, out_bp = self.aggregator.run(paths, graph, clusters, out_dict["dist_img"])

        ########################################################
        data_save = {
            "nodes": out_dict["nodes"],
            "edges_knn": out_dict["edges_knn"],
            "edges_lp": out_dict["edges_lp"],
            "edges_f": out_dict["edges_f"],
            "pred_weight": out_dict["pred_weight"],
            "pred_angle": out_dict["pred_angle"],
            "dist_img": out_dict["dist_img"],
            "pred_cls": node_cls,
            "graph": graph,
            "paths": paths_final,
            "int_list": out_int,
            "bp_list": out_bp,
        }

        ########################################################

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(15, 9))
            axs[0].imshow(mask_img, cmap="gray")
            axs[0].axis("off")
            for k, path in enumerate(paths_final.values()):
                path = np.array(path["points"])
                axs[0].plot(path[:, 1], path[:, 0], "o-", lw=3, label=k, color=cmap[k])

            #####
            axs[1].imshow(mask_img)
            axs[1].axis("off")
            cmap = plt.cm.get_cmap("jet")
            for node in graph.nodes():
                p = graph.nodes[node]["pos"]
                s = node_cls[node]
                axs[1].scatter(p[1], p[0], color=cmap(s), s=150, zorder=150)
                axs[1].text(p[1], p[0], str(node), fontsize=10, color="white", zorder=200, ha="center", va="center")

            for e0, e1 in graph.edges():
                pos_e0 = graph.nodes[e0]["pos"]
                pos_e1 = graph.nodes[e1]["pos"]
                axs[1].plot([pos_e0[1], pos_e1[1]], [pos_e0[0], pos_e1[0]], color="black", linewidth=1)

            plt.tight_layout()
            plt.show()

        return paths_final, out_int, out_bp, data_save


class LinkAnglePredictor:
    def __init__(self, checkpoint_link_pred):
        checkpoint_lp_path = checkpoint_link_pred

        state_lp = torch.load(checkpoint_lp_path, map_location=torch.device("cpu"))

        set_seeds(state_lp["seed"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("LinkAnglePredictor, device ", self.device)

        # MODEL
        self.model_lp = Network(state_lp)
        self.model_lp.load_state_dict(state_lp["model"])
        self.model_lp.to(self.device)
        self.model_lp.eval()

        # model number of parameters
        model_parameters = filter(lambda p: p.requires_grad, self.model_lp.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters: ", params)

        ##################
        self.graph_gen = GraphGenerationWh(debug=False)

        self.solver = SolverLPCheck()

    def preprocess_data(self, mask_img):
        graph_data = self.graph_gen.exec(mask_img, gt_data_flag=False)
        data = get_data_pyg(graph_data["nodes"], graph_data["edges_knn"], graph_data["dist_img"], path=None)
        return data

    @torch.no_grad()
    def predict(self, data: torch_geometric.data.Data):
        data = data.to(self.device)
        data.batch = torch.tensor([0] * data.num_nodes).type(torch.FloatTensor)
        out_lp = self.model_lp.predict(data)
        return (
            out_lp["pred_weight"].detach().cpu(),
            out_lp["pred_angle"].detach().cpu(),
            out_lp["mask_enc"].detach().cpu(),
            out_lp["node_feat"].detach().cpu(),
        )

    def run(self, mask_img):
        data_pyg = self.preprocess_data(mask_img)

        pred_weight, node_angle, mask_enc, node_feat = self.predict(data_pyg)

        out_solver = self.solver.run(data_pyg, node_angle, pred_weight)
        out_solver["mask_enc"] = mask_enc
        out_solver["node_feat"] = node_feat

        return data_pyg, out_solver

    def run_from_data_pyg(self, data_pyg):
        pred_weight, node_angle, mask_enc, node_feat = self.predict(data_pyg)
        out_solver = self.solver.run(data_pyg, node_angle, pred_weight)
        out_solver["mask_enc"] = mask_enc
        out_solver["node_feat"] = node_feat
        return out_solver

    def run_lp_only(self, data_pyg, lp_th=0.5):
        pred_weight, _ = self.predict(data_pyg)
        edges_knn = to_numpy(data_pyg.edge_index).T.squeeze()
        pred_w_np = pred_weight.squeeze().detach().cpu().numpy()
        lp_edge_indices = np.where(pred_w_np > lp_th)[0]
        return edges_knn[lp_edge_indices]


class SgClsPredictor:
    def __init__(self, checkpoint_cls):
        checkpoint_cls_path = checkpoint_cls

        state_cls = torch.load(checkpoint_cls_path, map_location=torch.device("cpu"))

        set_seeds(state_cls["seed"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("Using device ", self.device)
        print("SgClsPredictor, device ", self.device)

        # MODEL
        self.model_cls = SubGraphClassificator(state_cls)
        self.model_cls.load_state_dict(state_cls["model"])
        self.model_cls.to(self.device)
        self.model_cls.eval()

        # model number of parameters
        model_parameters = filter(lambda p: p.requires_grad, self.model_cls.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters: ", params)

        self.solver_sg = SolverSgCls()

    @torch.no_grad()
    def predict(self, data: torch_geometric.data.Data):
        data = data.to(self.device)
        data.batch = torch.tensor([0] * data.num_nodes).type(torch.FloatTensor)
        pred_cls = self.model_cls(data).sigmoid()
        return pred_cls

    def run(self, data_pyg, out_dict):
        dist = to_numpy(data_pyg.dist).squeeze()

        nodes = out_dict["nodes"]
        nodes_z = out_dict["node_feat"].detach().numpy()
        nodes_mask_enc = out_dict["mask_enc"].detach().numpy()
        graph = out_dict["graph"]
        dist_img = out_dict["dist_img"]

        ################

        data_list = [
            compute_data_subgraph(graph, dist, nodes, nodes_z, nodes_mask_enc, node_id=id) for id in range(len(nodes))
        ]
        data_is_valid, data_sg_list = zip(*data_list)
        data_is_valid = np.array(data_is_valid)
        data_sg_valid = [d for d in data_sg_list if d is not None]

        batch = Batch.from_data_list(data_sg_valid)
        batch = batch.to(self.device)
        pred = self.model_cls(batch).sigmoid().squeeze().detach().cpu().numpy()

        node_cls = np.zeros(len(nodes))
        node_cls[data_is_valid] = pred.squeeze()

        ################

        graph, clusters = self.solver_sg.run(graph, node_cls, dist_img)

        if False:
            fig = plt.figure(figsize=(10, 7))
            plt.imshow(dist_img)
            plt.axis("off")

            cmap = plt.cm.get_cmap("jet")
            for node in graph.nodes():
                node_pos = graph.nodes[node]["pos"]
                score = node_cls[node]
                plt.scatter(node_pos[1], node_pos[0], color=cmap(score), s=150, zorder=150)
                if True:
                    plt.text(
                        node_pos[1],
                        node_pos[0],
                        str(node),
                        fontsize=10,
                        color="white",
                        zorder=200,
                        ha="center",
                        va="center",
                    )

            for e0, e1 in graph.edges():
                pos_e0 = graph.nodes[e0]["pos"]
                pos_e1 = graph.nodes[e1]["pos"]
                plt.plot([pos_e0[1], pos_e1[1]], [pos_e0[0], pos_e1[0]], color="black", linewidth=3)

            plt.tight_layout()
            plt.show()

        return graph, node_cls, clusters


def cosine_sim_np(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_edge_from_ids(graph, n1, n2):
    dir = np.array(graph.nodes[n1]["pos"]) - np.array(graph.nodes[n2]["pos"])
    norm = np.linalg.norm(dir)
    dir = dir / norm
    return dir, norm
