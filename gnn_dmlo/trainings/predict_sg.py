import numpy as np
import torch, os, glob
from gnn_dmlo.model.network import SubGraphClassificator, Network
from gnn_dmlo.core.solver import SolverLPCheck
from gnn_dmlo.core.utils import set_seeds, to_numpy, create_nx_graph, compute_data_subgraph
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

############################################


class Predictor:
    def __init__(self, checkpoint_path, checkpoint_lp_path):
        self.device = "cpu"

        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        set_seeds(state["seed"])

        self.model = SubGraphClassificator(state)

        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.model.eval()

        # load model link pred
        state = torch.load(checkpoint_lp_path, map_location=torch.device("cpu"))
        self.model_lp = Network(state)
        self.model_lp.load_state_dict(state["model"])
        self.model_lp.eval()

        # solver
        self.solver = SolverLPCheck()

    @torch.no_grad()
    def exec(self, data):
        print(data)

        data.batch = torch.tensor([0] * data.x.shape[0]).type(torch.FloatTensor)
        out_pred = self.model_lp.predict(data)
        out_dict = self.solver.run(data, pred_angle=out_pred["pred_angle"], pred_weight=out_pred["pred_weight"])

        nodes = out_dict["nodes"]
        edges_mp = out_dict["edges_f"]
        edges_gt = to_numpy(data.edge_pos)
        dist_img = out_dict["dist_img"]
        nodes_z = out_pred["node_feat"].detach().numpy()
        nodes_mask_enc = out_pred["mask_enc"].detach().numpy()

        graph = create_nx_graph(nodes, edges_mp)

        preds = []
        for id, _ in enumerate(nodes):
            rv, data_sg = compute_data_subgraph(graph, dist_img, nodes, nodes_z, nodes_mask_enc, node_id=id)
            if rv is False:
                preds.append(0.0)
            else:
                out_cls = self.model(data_sg).sigmoid()
                preds.append(out_cls.item())

        plot_pyg_data(dist_img, nodes, edges_mp, preds)


def plot_pyg_data(dist_img, nodes, edges, preds):
    cmap = plt.get_cmap("jet")
    colors = [cmap(p) for p in preds]

    plt.imshow(dist_img)
    for it, node_pos in enumerate(nodes):
        plt.scatter(node_pos[1], node_pos[0], c=colors[it], s=50)

    for e0, e1 in edges:
        pos_e0, pos_e1 = nodes[e0], nodes[e1]
        plt.plot([pos_e0[1], pos_e1[1]], [pos_e0[0], pos_e1[0]], color="black", linewidth=1)

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    checkpoint_name = "nodecls_feasible-shape-85_20k.pth"
    checkpoint_lp = "linkpred_playful-sponge-91_136k.pth"
    dataset_name = "xxxx"

    script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(script_path, "checkpoints", checkpoint_name)
    ckpt_lp_path = os.path.join(script_path, "checkpoints", checkpoint_lp)
    dataset_path = os.path.join(script_path, "datasets", dataset_name)

    predictor = Predictor(ckpt_path, ckpt_lp_path)

    files = sorted(glob.glob(os.path.join(dataset_path, "*")))
    for f in files:
        data = torch.load(f)

        predictor.exec(data)
