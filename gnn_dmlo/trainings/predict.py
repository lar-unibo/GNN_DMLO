import numpy as np
import torch, torch_geometric, os
from gnn_dmlo.trainings.dataset_lp import GraphDatasetLinkPred
from gnn_dmlo.trainings.dataset_sg_cls import GraphDatasetNodeCls
from gnn_dmlo.model.network import Network, NodeClassificator
from gnn_dmlo.core.utils import set_seeds, to_numpy, create_nx_graph, line_from_angle
import matplotlib.pyplot as plt
import networkx as nx
from termcolor import cprint

np.set_printoptions(suppress=True)

############################################


class Predictor:
    def __init__(self, checkpoint_name):
        script_path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_path, f"checkpoints/{checkpoint_name}")

        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        dataset_path = os.path.join(script_path, "datasets", state["dataset_val_path"])

        set_seeds(state["seed"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device ", self.device)

        if "linkpred" in checkpoint_name:
            self.task = "linkpred"
            self.model = Network(state)
            dataset = GraphDatasetLinkPred(dataset_path)
        elif "nodecls" in checkpoint_name:
            self.task = "nodecls"
            self.model = NodeClassificator(state)
            dataset = GraphDatasetNodeCls(dataset_path)
        else:
            raise ValueError("")

        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.model.eval()

        self.loader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)

    @torch.no_grad()
    def exec(self):
        for data in self.loader:
            data = data.to(self.device)

            if self.task == "linkpred":
                out = self.model.predict(data)
                pred_weight = out["pred_weight"]
                pred_angle = out["pred_angle"]
                plot_pyg_data(data, edges_w=pred_weight, node_dir=pred_angle)

            elif self.task == "nodecls":
                out_cls = self.model(data)
                if out_cls.shape[-1] == 1:
                    out_cls = out_cls.sigmoid()

                plot_pyg_data(data, node_cls=out_cls)


############################################


def draw_graph(graph, axis, node_size=200, font_size=8, colors=None):
    plt.sca(axis)

    if colors is None:
        colors = "lightblue"

    nx.draw(
        graph,
        pos={k: (n["pos"][1], n["pos"][0]) for k, n in graph.nodes(data=True)},
        node_size=node_size,
        node_color=colors,
    )
    if font_size > 0:
        for k, n in graph.nodes(data=True):
            axis.text(
                n["pos"][1], n["pos"][0], k, fontsize=font_size, ha="center", va="center", color="white", weight="bold"
            )
    axis.axis("off")


def plot_pyg_data(data, edges_w=None, node_dir=None, node_cls=None):
    mask = to_numpy(data.mask).squeeze()
    edges_init = to_numpy(data.edge_index).T
    edges_gt = to_numpy(data.edge_pos)
    nodes = [(int(n[0] * mask.shape[0]), int(n[1] * mask.shape[1])) for n in to_numpy(data.x)]

    if edges_w is None:
        edges_w = to_numpy(data.pred_weight).squeeze()

    if node_dir is None:
        node_dir = to_numpy(data.pred_angle).squeeze()
    else:
        node_dir = to_numpy(node_dir).squeeze()

    if node_cls is not None:
        node_cls = to_numpy(node_cls).squeeze()

    # PRINT
    classes = ["normal", "branch", "intersection"]
    print_colors = ["blue", "red", "green"]
    for i in range(len(nodes)):
        indices_0 = np.where(edges_init[:, 0] == i)[0]
        indices_1 = np.where(edges_init[:, 1] == i)[0]
        indices = np.concatenate([indices_0, indices_1])
        node_edges = edges_init[indices].tolist()
        node_edges_w = edges_w[indices].tolist()

        if node_cls is not None:
            value = node_cls[i]
            label = np.argmax(value)
            cprint(f"Node {i} -> {value} , {classes[label]} ", print_colors[label])
        else:
            cprint(f"Node {i}", "yellow")
        print(list(zip(node_edges, node_edges_w)))

    # graphs
    G_input = create_nx_graph(nodes, edges_init)
    G_gt = create_nx_graph(nodes, edges_gt)

    #####################################################

    if node_cls is not None:
        if node_cls.ndim == 1:
            cmap = plt.cm.get_cmap("jet")
            colors = [cmap(p) for p in node_cls]
        else:
            colors = np.argmax(node_cls, axis=1)
            colors_values = np.max(node_cls, axis=1)
            cmaps = [plt.cm.get_cmap("Blues"), plt.cm.get_cmap("Reds"), plt.cm.get_cmap("Greens")]
            colors = [cmaps[c](p) for c, p in zip(colors, colors_values)]
    else:
        colors = None

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    axs[0].imshow(mask)
    draw_graph(G_input, axis=axs[0])
    axs[0].set_title("G_input")

    axs[1].imshow(mask)
    draw_graph(G_gt, axis=axs[1], colors=colors)

    for i in range(len(node_dir)):
        angle = np.deg2rad(np.argmax(node_dir[i]))
        pos = nodes[i]
        x, y = line_from_angle(angle, pos=pos, length=3)
        axs[1].plot(y, x, linewidth=5, c="gray")

    axs[1].set_title("G_gt")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    checkpoint_name = "nodecls_fiery-serenity-195.pth"
    predictor = Predictor(checkpoint_name)

    predictor.exec()
