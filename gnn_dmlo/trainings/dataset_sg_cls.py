import os, glob, torch
from tqdm import tqdm
from gnn_dmlo.model.network import Network
from gnn_dmlo.core.utils import create_new_folder, create_nx_graph, to_numpy, compute_data_subgraph
import numpy as np
import cv2

from gnn_dmlo.core.solver import SolverLPCheck
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from shapely import geometry


class GraphPreProcessingCls:
    def __init__(self, dataset_path, checkpoint_lp, max_samples=5000, excl_hops=3):
        self.dataset_path = dataset_path
        self.max_samples = max_samples
        self.excl_hops = excl_hops

        dirname = os.path.dirname(dataset_path)
        basename = os.path.basename(dataset_path)

        # load model link pred
        state = torch.load(checkpoint_lp, map_location=torch.device("cpu"), weights_only=False)
        self.model = Network(state)
        self.model.load_state_dict(state["model"])
        self.model.eval()

        # solver
        self.solver = SolverLPCheck()

        # OUT NAME
        lp_name = os.path.basename(checkpoint_lp).split(".")[0].split("_")[-1]
        out_name = basename + f"_SG_{lp_name}_exclhops{self.excl_hops}"

        # folder where to save graphs
        self.graphs_path = os.path.join(dirname, out_name)
        create_new_folder(self.graphs_path)

        # folder input graphs
        self.graphs_path_lp = os.path.join(dirname, basename)
        self.preprocess()

    def preprocess(self) -> None:
        counter = self.max_samples
        for f in tqdm(sorted(glob.glob(os.path.join(self.graphs_path_lp, "*")))):
            try:
                name_without_ext = f.split("/")[-1].split(".")[0]
                data = torch.load(f, weights_only=False)
                data_new_list = self.get_data_subgraphs(data)
                for i, data_new in enumerate(data_new_list):
                    torch.save(data_new, os.path.join(self.graphs_path, name_without_ext + f"_{i}.pt"))
                    counter -= 1

            except Exception as e:
                print(f"Error in {f}: {e}")
                continue

            if counter < 0:
                break

    def compute_intersections(self, graph_gt):
        edges_gt = np.array(list(graph_gt.edges))
        positions = []
        combs = list(itertools.combinations(edges_gt.tolist(), 2))
        for c1, c2 in combs:
            if c1[0] != c2[0] and c1[0] != c2[1] and c1[1] != c2[0] and c1[1] != c2[1]:
                pos10 = graph_gt.nodes[c1[0]]["pos"]
                pos11 = graph_gt.nodes[c1[1]]["pos"]
                pos20 = graph_gt.nodes[c2[0]]["pos"]
                pos21 = graph_gt.nodes[c2[1]]["pos"]
                line1 = geometry.LineString([pos10, pos11])
                line2 = geometry.LineString([pos20, pos21])
                if line1.intersects(line2):
                    int_pos = line1.intersection(line2)
                    int_pos = [p for p in int_pos.coords]
                    # print("intersection: ", int_pos)
                    positions.extend(int_pos)
        return positions

    def mask_branchpoints(self, dist_img, branch_points):
        mask_bp = np.zeros_like(dist_img)
        for bp in branch_points:
            x_values = np.arange(bp[1] - 10, bp[1] + 10)
            y_values = np.arange(bp[0] - 10, bp[0] + 10)
            dist_values = []
            for x in x_values:
                for y in y_values:
                    if x > 0 and x < mask_bp.shape[1] and y > 0 and y < mask_bp.shape[0]:
                        dist_values.append(dist_img[int(y), int(x)])

            if len(dist_values) == 0:
                continue

            dist_th = max(np.max(dist_values), 5) * 3
            cv2.circle(mask_bp, tuple([int(bp[1]), int(bp[0])]), int(dist_th), 255, -1)
        return mask_bp

    def get_data_subgraphs(self, data):
        data.batch = torch.tensor([0] * data.x.shape[0]).type(torch.FloatTensor)
        out_pred = self.model.predict(data)
        out_dict = self.solver.run(data, pred_angle=out_pred["pred_angle"], pred_weight=out_pred["pred_weight"])

        nodes = out_dict["nodes"]
        edges_mp = out_dict["edges_f"]
        edges_gt = to_numpy(data.edge_pos)
        dist_img = out_dict["dist_img"]
        nodes_z = out_pred["node_feat"].detach().numpy()
        nodes_mask_enc = out_pred["mask_enc"].detach().numpy()

        bps = to_numpy(data.bps)
        bp_mask = self.branch_points_ints_mask(nodes, edges_gt, bps, dist_img)

        sg_data_list = self.subgraphs_type_1(dist_img, nodes, edges_mp, nodes_z, nodes_mask_enc, bp_mask, plot=True)

        return sg_data_list

    def branch_points_ints_mask(self, nodes, edges_gt, bps, dist_img):
        graph_gt = create_nx_graph(nodes, edges_gt)
        int_pos = self.compute_intersections(graph_gt)
        positive_pos = np.concatenate((bps, int_pos), axis=0)
        return self.mask_branchpoints(dist_img, positive_pos)

    def subgraphs_type_1(self, dist_img, nodes, edges_mp, nodes_z, nodes_mask_enc, bp_mask, plot=False):
        ###################
        graph_mp = create_nx_graph(nodes, edges_mp)
        high_deg_nodes = [n for n, d in graph_mp.degree if d > 2]

        sg_data_list = []
        nodes_label_1 = []
        for node_id in high_deg_nodes:
            pos = nodes[node_id]

            vm = bp_mask[int(pos[0]), int(pos[1])]
            label = 1 if vm == 255 else 0
            if label == 0:
                continue

            rv, data_new = compute_data_subgraph(graph_mp, dist_img, nodes, nodes_z, nodes_mask_enc, node_id=node_id)
            if rv is False:
                continue

            data_new.label = torch.tensor(label).type(torch.LongTensor)
            sg_data_list.append(data_new)
            nodes_label_1.append(node_id)

        node_excluded = []
        for it, node_pos in enumerate(nodes):
            if it in nodes_label_1:
                continue

            min_dist = np.inf
            for n in nodes_label_1:
                dist = len(nx.shortest_path(graph_mp, source=n, target=it))
                if dist < min_dist:
                    min_dist = dist

            if min_dist <= self.excl_hops:
                node_excluded.append(it)

        nodes_label_0 = []
        for node_id in high_deg_nodes:
            if node_id in node_excluded or node_id in nodes_label_1:
                continue

            pos = nodes[node_id]

            vm = bp_mask[int(pos[0]), int(pos[1])]
            label = 1 if vm == 255 else 0
            if label == 1:
                continue

            rv, data_new = compute_data_subgraph(graph_mp, dist_img, nodes, nodes_z, nodes_mask_enc, node_id=node_id)
            if rv is False:
                continue
            data_new.label = torch.tensor(label).type(torch.LongTensor)
            sg_data_list.append(data_new)
            nodes_label_0.append(node_id)

            if len(nodes_label_0) > len(nodes_label_1):
                break

        ############
        # random node with deg 2
        deg2_nodes = [n for n, d in graph_mp.degree if d == 2]
        for node_id in deg2_nodes:
            nn = list(graph_mp.neighbors(node_id))

            if nn[0] in nodes_label_1 or nn[1] in nodes_label_1:
                continue

            if nn[0] in node_excluded or nn[1] in node_excluded:
                continue

            pos = nodes[node_id]
            vm = bp_mask[int(pos[0]), int(pos[1])]
            label = 1 if vm == 255 else 0
            if label == 1:
                continue

            rv, data_new = compute_data_subgraph(graph_mp, dist_img, nodes, nodes_z, nodes_mask_enc, node_id=node_id)
            if rv is False:
                continue

            data_new.label = torch.tensor(label).type(torch.LongTensor)
            sg_data_list.append(data_new)
            nodes_label_0.append(node_id)

            if len(nodes_label_0) > len(nodes_label_1):
                break

        # print("lens: ", len(nodes_label_0), len(nodes_label_1))

        if plot:
            plt.imshow(dist_img)
            for it, node_pos in enumerate(nodes):
                if it in nodes_label_1:
                    color = "tab:green"
                elif it in node_excluded:
                    color = "tab:orange"
                elif it in nodes_label_0:
                    color = "tab:blue"
                else:
                    color = "tab:red"

                plt.scatter(node_pos[1], node_pos[0], c=color, s=50)

            for e0, e1 in graph_mp.edges():
                pos_e0, pos_e1 = nodes[e0], nodes[e1]
                plt.plot([pos_e0[1], pos_e1[1]], [pos_e0[0], pos_e1[0]], color="black", linewidth=1)

            plt.show()

        return sg_data_list


class GraphDatasetNodeCls(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str) -> None:
        self.files = sorted(glob.glob(os.path.join(dataset_path, "*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])


if __name__ == "__main__":

    checkpoint_path = "/home/alessio/Downloads/gnn_dlo/checkpoints/linkpred.pth"
    path = "../datasets/example_synthetic_dataset_LP_augmentation"
    g = GraphPreProcessingCls(checkpoint_lp=checkpoint_path, dataset_path=path, max_samples=1000)
