import os, cv2, glob
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gnn_dmlo.core.core import LinkAnglePredictor


def plot_graph(nodes, edges, ax, node_size=50, colors=None):
    for it, n in enumerate(nodes):
        color = colors[it] if colors is not None else "tab:blue"
        ax.scatter(n[1], n[0], color=color, s=node_size, zorder=150)
        # ax.text(n[1], n[0], str(it), fontsize=10, color="white", zorder=200, ha="center", va="center")

    for e0, e1 in edges:
        pos_e0, pos_e1 = nodes[e0], nodes[e1]
        ax.plot([pos_e0[1], pos_e1[1]], [pos_e0[0], pos_e1[0]], color="black", linewidth=1)


def plot_output(mask, out_dict, save_path=None):
    graph = nx.Graph()
    graph.add_nodes_from({i: {"pos": n} for i, n in enumerate(out_dict["nodes"])})
    graph.add_edges_from(out_dict["edges_f"])

    deg = dict(graph.degree)
    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(deg[i]) for i in range(len(deg))]

    nodes = out_dict["nodes"]
    edges_lp = out_dict["edges_lp"]
    edges_f = out_dict["edges_f"]
    node_angle = out_dict["pred_angle"].detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    ax1.imshow(mask)
    plot_graph(nodes, edges_lp, ax1, node_size=50)

    ax2.imshow(mask)
    plot_graph(nodes, edges_f, ax2, colors=colors)

    for it, node in enumerate(nodes):
        angle = np.argmax(node_angle[it])
        angle = np.deg2rad(angle)
        dir = np.array([np.sin(angle), np.cos(angle)])
        dir = dir / np.linalg.norm(dir)
        p1 = node + dir * 5
        p2 = node - dir * 5
        ax2.plot([p1[1], p2[1]], [p1[0], p2[0]], color="green", linewidth=2)

    ax1.set_title("LP")
    ax2.set_title("F")
    ax1.axis("off")
    ax2.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    ############################################

    CHECKPOINT_LP_PATH = "../checkpoints/linkpred.pth"
    SAMPLES_PATH = "../example_real_samples"

    ############################################

    p = LinkAnglePredictor(CHECKPOINT_LP_PATH)

    for mask_path in sorted(glob.glob(os.path.join(SAMPLES_PATH, "*.png"))):

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 255

        data_pyg, out_dict = p.run(mask)
        plot_output(mask, out_dict)
