import numpy as np
import networkx as nx
import itertools, cv2, torch_geometric, torch
import matplotlib.pyplot as plt
from gnn_dmlo.core.utils import to_numpy, create_nx_graph_angles
from scipy.spatial.distance import cdist


def node_dir_from_angles(pred_angle: torch.Tensor):
    node_angle = torch.deg2rad(torch.argmax(pred_angle, dim=1))
    return torch.cat((torch.sin(node_angle).unsqueeze(1), torch.cos(node_angle).unsqueeze(1)), dim=1)


def cosine_sim_tensor(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def cosine_sim_np(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_edge_from_ids(graph, n1, n2):
    dir = np.array(graph.nodes[n1]["pos"]) - np.array(graph.nodes[n2]["pos"])
    return dir / np.linalg.norm(dir)


def filter_graph_edges_from_mask(graph, mask):
    # mask_large = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    nodes = np.array([v["pos"] for _, v in graph.nodes(data=True)])
    edges = np.array(list(graph.edges()))

    new_edges = filter_edges_from_mask(nodes, edges, mask)

    g = nx.Graph()
    g.add_nodes_from(graph.nodes(data=True))
    g.add_edges_from(new_edges)
    return g


def filter_edges_from_mask(nodes, edges, mask):
    mask_large = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

    nodes_0 = nodes[edges[:, 0]]
    nodes_1 = nodes[edges[:, 1]]

    distances = np.linalg.norm(nodes_0 - nodes_1, axis=1)
    indeces = np.where(distances > np.mean(distances))[0]

    mid_points_to_test = ((nodes_0[indeces] + nodes_1[indeces]) / 2).astype(int)
    indeces_zero = np.where(mask_large[mid_points_to_test[:, 0], mid_points_to_test[:, 1]] == 0)[0]

    mask = np.ones(edges.shape[0], bool)
    mask[indeces[indeces_zero]] = 0
    return edges[mask]


class SolverLPCheck:
    def run(
        self,
        data_pyg: torch_geometric.data.Data,
        pred_angle: torch.Tensor,
        pred_weight: torch.Tensor,
        lp_th: float = 0.2,
    ):
        nodes = to_numpy(data_pyg.x)
        mask = to_numpy(data_pyg.mask).squeeze()
        dist_img = to_numpy(data_pyg.dist).squeeze()
        nodes = nodes * mask.shape

        node_dir = node_dir_from_angles(pred_angle)

        edges_knn = to_numpy(data_pyg.edge_index).T.squeeze()

        pred_w_np = pred_weight.squeeze().detach().cpu().numpy()
        lp_edge_indices = np.where(pred_w_np > lp_th)[0]
        edges_lp = edges_knn[lp_edge_indices]

        G = create_nx_graph_angles(nodes, edges_lp, to_numpy(node_dir))

        G = self.check_edges_fast(G, dist_img, dist_gain=2, similarity_th=0.1)
        G = filter_graph_edges_from_mask(G, mask)

        G = self.augment_edges_node_single(G, edges_knn)
        G = filter_graph_edges_from_mask(G, mask)

        edges_f = np.array(list(G.edges()))

        return {
            "nodes": nodes,
            "edges_knn": edges_knn,
            "edges_lp": edges_lp,
            "edges_f": edges_f,
            "pred_weight": pred_weight,
            "pred_angle": pred_angle,
            "dist_img": dist_img,
            "graph": G,
        }

    def augment_edges_node_single(self, G, edges_knn):
        nodes_single = [n for n in G.nodes if G.degree(n) == 1]
        for n in nodes_single:
            neigh = list(G.neighbors(n))[0]
            dir = compute_edge_from_ids(G, n, neigh)

            neigh_knn = [e for e in edges_knn if e[0] == n or e[1] == n]
            neigh_knn = [e[0] if e[0] != n else e[1] for e in neigh_knn]
            # print(n, neigh, neigh_knn)

            scores = []
            for n_knn in neigh_knn:
                edge_knn = compute_edge_from_ids(G, n, n_knn)
                s = cosine_sim_np(dir, edge_knn)
                scores.append(s)

            min_idx = np.argmin(scores)
            min_score = scores[min_idx]
            if min_score < -0.5:
                G.add_edge(n, neigh_knn[min_idx])
                # print("add edge", n, neigh_knn[min_idx], min_score)

        return G

    def check_edges_fast(self, G, dist_img, dist_gain=2, similarity_th=0.2):
        # get edges and nodes from the graph
        edges = np.array(list(G.edges()))
        nodes = np.array([v["pos"] for _, v in G.nodes(data=True)])

        nodes_id = np.array([k for k, _ in G.nodes(data=True)])
        nodes_dir = np.array([v["angle"] for _, v in G.nodes(data=True)])

        # dist img values for each node
        dist_img_nodes = dist_img[nodes[:, 0].astype(int), nodes[:, 1].astype(int)]
        dist_img_nodes_r = np.repeat(dist_img_nodes[:, np.newaxis], edges.shape[0], axis=1)

        # repeat edges and nodes to have the same shape
        edges_r = np.repeat(edges[np.newaxis, :, :], nodes.shape[0], axis=0)
        nodes_r = np.repeat(nodes[:, np.newaxis, :], edges.shape[0], axis=1)
        nodes_id_r = np.repeat(nodes_id[:, np.newaxis], edges.shape[0], axis=1)

        # get the nodes of the edges
        nodes_e0 = nodes[edges[:, 0]]
        nodes_e1 = nodes[edges[:, 1]]

        # repeat the nodes of the edges to have the same shape
        nodes_e0r = np.repeat(nodes_e0[np.newaxis, :, :], nodes.shape[0], axis=0)
        nodes_e1r = np.repeat(nodes_e1[np.newaxis, :, :], nodes.shape[0], axis=0)

        # distance between nodes and edge segments
        v1 = nodes_r - nodes_e0r
        v2 = nodes_e1r - nodes_e0r
        dot_product = np.multiply(v1, v2).sum(axis=2)
        segment_length = np.linalg.norm(v2, axis=2)
        projection = dot_product / segment_length**2

        # cosine similarity between nodes orientation and edge segments direction
        S = cosine_sim_tensor(
            torch.from_numpy(nodes_dir).type(torch.FloatTensor),
            torch.from_numpy(nodes_e1 - nodes_e0).type(torch.FloatTensor),
        )
        S = S.detach().cpu().numpy()

        # masks
        mask_proj = np.logical_and(projection > 0, projection < 1)
        mask_dist = np.linalg.norm(v1 - projection[:, :, np.newaxis] * v2, axis=2) < dist_img_nodes_r * dist_gain
        mask_S = np.fabs(S) > similarity_th
        mask_combined = np.logical_and(np.logical_and(mask_proj, mask_dist), mask_S)
        edges_to_process = edges_r[mask_combined, :]
        nodes_to_process = nodes_id_r[mask_combined]
        nodes_proj = projection[mask_combined]

        edges_tuples = np.unique(edges_to_process, axis=0)
        to_process = {tuple(edge): [] for edge in edges_tuples}
        for i in range(nodes_to_process.shape[0]):
            edge = tuple(edges_to_process[i])
            node = nodes_to_process[i]
            if node != edge[0] and node != edge[1]:
                to_process[edge].append((node, nodes_proj[i]))

        edges_to_add = []
        edges_to_remove = []
        for edge, values in to_process.items():
            if len(values) == 0:
                continue

            values = sorted(values, key=lambda x: x[1])
            v_edges = [n for n, _ in values]
            edges_to_add.extend([(v_edges[i], v_edges[i + 1]) for i in range(len(v_edges) - 1)])
            edges_to_add.extend([(v_edges[-1], edge[1]), (edge[0], v_edges[0])])
            edges_to_remove.append(edge)

        G.remove_edges_from(edges_to_remove)
        G.add_edges_from(edges_to_add)
        return G


class SolverSgCls:
    def __init__(self, th_cls=0.3):
        self.solver_lp = SolverLPCheck()
        self.th_cls = th_cls
        self.solver_int = SolverIntersections()

    def run(self, graph, node_cls, dist):
        mask = np.zeros_like(dist)
        mask[dist > 0] = 1

        graph_zero = self.run_zero_points(graph, node_cls)
        graph_one, clusters = self.run_one_points(graph_zero, dist)

        # graph_int = self.solver_sg.solve_intersections(graph_one)

        graph_one = filter_graph_edges_from_mask(graph_one, mask)

        return graph_one, clusters

    def solve_intersections(self, graph):
        graph = self.solve_small_cycles(graph)
        return self.solver_int.run(graph)

    def solve_small_cycles(self, graph):
        cycles = nx.cycle_basis(graph)
        cycles = [c for c in cycles if len(c) < 8]
        for cycle in cycles:
            deg3_node = [n for n in cycle if graph.degree(n) == 3]
            if len(deg3_node) == 1:  # apply simple heuristic
                deg2_node = [n for n in cycle if graph.degree(n) == 2]
                if len(deg2_node) + len(deg3_node) == len(cycle):
                    # remove longest edge of deg3 node
                    neigh_deg3 = list(graph.neighbors(deg3_node[0]))
                    edges = [
                        np.array(graph.nodes[deg3_node[0]]["pos"]) - np.array(graph.nodes[n]["pos"])
                        for n in neigh_deg3
                    ]
                    edges_len = [np.linalg.norm(e) for e in edges]
                    longest_edge_idx = np.argmax(edges_len)
                    graph.remove_edge(deg3_node[0], neigh_deg3[longest_edge_idx])

            if len(deg3_node) == 2:
                # remove edge between the two deg3 nodes
                if graph.has_edge(deg3_node[0], deg3_node[1]):
                    graph.remove_edge(deg3_node[0], deg3_node[1])
                    # check connectivity
                    if nx.has_path(graph, deg3_node[0], deg3_node[1]):
                        continue
                    else:
                        graph.add_edge(deg3_node[0], deg3_node[1])

        return graph

    def run_zero_points(self, graph, preds):
        graph_out = graph.copy()
        high_deg_nodes = [n for n in graph_out.nodes if graph_out.degree(n) > 2]

        for node in high_deg_nodes:
            pred = preds[node]
            if pred > self.th_cls:
                continue

            edges_node = graph_out.edges(node)
            combs = list(itertools.combinations(edges_node, 2))
            scores = []
            for e0, e1 in combs:
                e0 = e0[0] if e0[0] != node else e0[1]
                e1 = e1[0] if e1[0] != node else e1[1]

                edge0 = compute_edge_from_ids(graph_out, e0, node)
                edge1 = compute_edge_from_ids(graph_out, node, e1)
                edge0 = edge0 / np.linalg.norm(edge0)
                s = cosine_sim_np(edge0, edge1)
                scores.append(s)

            if len(scores) > 0:
                best_idx = np.argmax(scores)
                best_comb = combs[best_idx]
                nodes_s = [best_comb[0][0], best_comb[0][1], best_comb[1][0], best_comb[1][1]]
                nodes_s = [n for n in nodes_s if n != node]

                edges_to_remove = [e for e in edges_node if e[0] not in nodes_s and e[1] not in nodes_s]
                graph_out.remove_edges_from(edges_to_remove)

        return graph_out

    def get_nodes_having_longest_edges(self, graph, node):
        # from a node with 2+ edges, get all the edgees excluding the two shortest
        edges_len = []
        nn = list(graph.neighbors(node))
        node_pos = np.array(graph.nodes[node]["pos"])
        for neigh in nn:
            edges_len.append(np.linalg.norm(node_pos - np.array(graph.nodes[neigh]["pos"])))
        return np.array(nn)[np.argsort(edges_len)][2:]

    def get_nodes_having_less_smooth_edge(self, graph, node):
        nn = list(graph.neighbors(node))
        pairs = list(itertools.combinations(nn, 2))
        scores = []
        for n1, n2 in pairs:
            edge1 = compute_edge_from_ids(graph, n1, node)
            edge2 = compute_edge_from_ids(graph, node, n2)
            edge1 = edge1 / np.linalg.norm(edge1)
            edge2 = edge2 / np.linalg.norm(edge2)
            s = cosine_sim_np(edge1, edge2)
            scores.append(s)

        best_pair = pairs[np.argmax(scores)]
        other_nodes = [n for n in nn if n not in best_pair]
        return other_nodes

    def run_one_points(self, graph, dist_img):
        graph_out = graph.copy()

        # cluster nodes having degree > 2
        clusters_dict = self.nodes_one_clustering(graph_out, dist_img)

        # self.plot_graph_and_cluster(graph_out, cluster_graph=None)

        # process each cluster
        for k, cluster in clusters_dict.items():
            # print("xxxxxxxxxxxxxxxxxxxxxxxx")
            # cluster as separate subgraph
            all_cluster_nodes = cluster["nodes_ext"] + cluster["nodes_nn"]
            cluster_edges = list(graph_out.edges(all_cluster_nodes))
            cluster_graph = graph_out.edge_subgraph(cluster_edges)

            # deg 1 node to guarantee connectivity
            node_1_graph = [n for n in cluster_graph.nodes() if cluster_graph.degree(n) == 1]
            combination_paths = list(itertools.combinations(node_1_graph, 2))
            # print("cluster ", k, cluster["nodes_ext"], node_1_graph)

            # get all edges to remove -> each node can have only 2 edges
            cluster_graph_copy = cluster_graph.copy()
            for n in cluster_graph.nodes():
                if cluster_graph_copy.degree(n) > 2:
                    # nn = self.get_nodes_having_longest_edges(cluster_graph, n)
                    nn = self.get_nodes_having_less_smooth_edge(cluster_graph_copy, n)
                    cluster_graph_copy.remove_edges_from([(n, neigh) for neigh in nn])

            for _ in range(10):
                # add edges between nodes having degree 1 -> recovering connectivity of the cluster
                node_1_inside = [n for n in cluster_graph_copy.nodes() if cluster_graph_copy.degree(n) == 1]
                node_1_inside = [n for n in node_1_inside if n not in node_1_graph]
                # print("-- node_1_inside", node_1_inside)

                if len(node_1_inside) == 0:
                    break

                n1 = node_1_inside[0]
                pos_n1 = np.array(graph_out.nodes[n1]["pos"])

                if True:
                    # connect to the closest node of the cluster forward
                    nodes_to_test = [n for n in cluster_graph_copy.nodes() if n != n1]

                    neigh_1 = list(cluster_graph_copy.neighbors(n1))[0]
                    neigh_1_pos = np.array(graph_out.nodes[neigh_1]["pos"])

                    forward_dir = pos_n1 - neigh_1_pos
                    forward_dir = forward_dir / np.linalg.norm(forward_dir)

                    if len(nodes_to_test) > 0:
                        pos = np.array([graph_out.nodes[n]["pos"] for n in nodes_to_test])
                        dirs = pos - pos_n1
                        dirs = dirs / np.linalg.norm(dirs, axis=1)[:, np.newaxis]
                        scores = np.dot(dirs, forward_dir)
                        nodes_to_test_forward = [nodes_to_test[i] for i, score in enumerate(scores) if score > 0]
                        # print("n1", n1, "nodes_to_test_forward", nodes_to_test_forward)
                        if len(nodes_to_test_forward) > 0:
                            pos = np.array([graph_out.nodes[n]["pos"] for n in nodes_to_test_forward])
                            idx = np.argmin(np.linalg.norm(pos - pos_n1, axis=1))
                            closest_node = nodes_to_test_forward[idx]
                            cluster_graph_copy.add_edge(n1, closest_node)
                            # print("add edge", n1, closest_node)

            # final check connectivity
            path_check = [nx.has_path(cluster_graph, p[0], p[1]) for p in combination_paths]
            if np.any(path_check) == False:
                print(f"cluster {k} not connected!!!!")
                print("path_check", path_check)

            cluster_edges_up = list(cluster_graph_copy.edges())
            graph_out.remove_edges_from(cluster_edges)
            graph_out.add_edges_from(cluster_edges_up)

        return graph_out, clusters_dict

    def plot_graph_and_cluster(self, graph, cluster_graph=None):
        for n in graph.nodes():
            plt.scatter(graph.nodes[n]["pos"][1], graph.nodes[n]["pos"][0], color="tab:blue", s=100)
            plt.text(
                graph.nodes[n]["pos"][1],
                graph.nodes[n]["pos"][0],
                str(n),
                fontsize=8,
                color="white",
                ha="center",
                va="center",
            )
        for e in graph.edges():
            pos0 = graph.nodes[e[0]]["pos"]
            pos1 = graph.nodes[e[1]]["pos"]
            plt.plot([pos0[1], pos1[1]], [pos0[0], pos1[0]], color="black", linewidth=1)

        if cluster_graph is not None:
            for n in cluster_graph.nodes():
                plt.scatter(cluster_graph.nodes[n]["pos"][1], cluster_graph.nodes[n]["pos"][0], color="tab:red", s=100)
                plt.text(
                    cluster_graph.nodes[n]["pos"][1],
                    cluster_graph.nodes[n]["pos"][0],
                    str(n),
                    fontsize=8,
                    color="white",
                    ha="center",
                    va="center",
                )
            for e in cluster_graph.edges():
                pos0 = cluster_graph.nodes[e[0]]["pos"]
                pos1 = cluster_graph.nodes[e[1]]["pos"]
                plt.plot([pos0[1], pos1[1]], [pos0[0], pos1[0]], color="tab:red", linewidth=1)

        plt.tight_layout()
        plt.show()

    def nodes_one_clustering(self, graph, dist_img):
        # clusters
        nodes = np.array([v["pos"] for _, v in graph.nodes(data=True)])
        high_deg_nodes = [n for n in graph.nodes if graph.degree(n) > 2]

        clusters = []
        for node_id in high_deg_nodes:
            nn = list(graph.neighbors(node_id))
            nn_high_deg = [n for n in nn if n in high_deg_nodes]
            nn_high_deg.append(node_id)
            clusters.append(nn_high_deg)

        for it, cluster in enumerate(clusters):
            for cluster2 in clusters[it + 1 :]:
                if len(set(cluster).intersection(set(cluster2))) > 0:
                    cluster.extend(cluster2)

        clusters_merged = []
        for cluster in clusters:
            skip = False
            for cluster_m in clusters_merged:
                if len(set(cluster).intersection(set(cluster_m))) > 0:
                    skip = True

            if not skip:
                cluster = list(set(cluster))
                clusters_merged.append(cluster)

        clusters_dict = {}
        for it, cluster in enumerate(clusters_merged):
            all_neigh = []
            for n in cluster:
                all_neigh.extend(list(graph.neighbors(n)))
            all_neigh = list(set(all_neigh))
            all_neigh = [n for n in all_neigh if n not in cluster]

            ext_cluster = []
            all_neigh_updated = all_neigh.copy()
            for neigh in all_neigh:
                nn = [n for n in list(graph.neighbors(neigh)) if n not in cluster]
                if len(nn) == 0:
                    ext_cluster.append(neigh)
                    all_neigh_updated.remove(neigh)
            ext_cluster.extend(cluster)

            # center
            nodes_c = nodes[cluster].astype(int)
            center = np.mean(nodes_c, axis=0).astype(int)

            # peak
            dist_values = dist_img[nodes_c[:, 0], nodes_c[:, 1]]
            peak_id = cluster[np.argmax(dist_values)]
            peak = nodes_c[np.argmax(dist_values)]

            clusters_dict[it] = {
                "nodes": cluster,
                "nodes_ext": ext_cluster,
                "nodes_nn": all_neigh_updated,
                "center_pos": center,
                "peak_pos": peak,
                "peak_id": peak_id,
            }

        return clusters_dict

    def plot_with_cluster(self, graph_out, clusters_dict, dist_img, sections=None):
        nodes = np.array([v["pos"] for _, v in graph_out.nodes(data=True)])

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(dist_img)

        cmap = plt.cm.get_cmap("tab20", len(clusters_dict) + 1)
        for it, n in enumerate(nodes):
            idx = None
            for c_it, cluster in clusters_dict.items():
                if it in cluster["nodes"]:
                    idx = c_it
                    break

            if idx is None:
                c = cmap(len(clusters_dict))
            else:
                c = cmap(idx)

            plt.scatter(n[1], n[0], color=c, s=150, zorder=150)
            plt.text(n[1], n[0], str(it), fontsize=10, color="white", zorder=200, ha="center", va="center")

        for it, cluster in clusters_dict.items():
            center = cluster["center_pos"]
            peak = cluster["peak_pos"]
            plt.scatter(center[1], center[0], color=cmap(it), s=150, zorder=150, marker="*")
            plt.scatter(peak[1], peak[0], color=cmap(it), s=150, zorder=150, marker="x")

        new_edges = list(graph_out.edges())
        for e0, e1 in new_edges:
            pos_e0, pos_e1 = nodes[e0], nodes[e1]
            plt.plot([pos_e0[1], pos_e1[1]], [pos_e0[0], pos_e1[0]], color="black", linewidth=1)

        if sections is not None:
            for s in sections:
                pos = [graph_out.nodes[n]["pos"] for n in s]
                pos = np.array(pos)
                plt.plot(pos[:, 1], pos[:, 0], color="red", linewidth=2, zorder=100)

        plt.tight_layout()
        plt.show()


class SolverIntersections:
    def run(self, graph, int_bp=True, int_pure=True):
        graph = graph.copy()

        #############
        if int_pure:
            intersection_nodes = [n for n in graph.nodes if graph.degree(n) == 4]
            print("intersection_nodes", intersection_nodes)
            for node in intersection_nodes:
                nn = list(graph.neighbors(node))
                possible_edges = list(itertools.combinations(nn, 2))
                nodes_test_dict = {n: compute_edge_from_ids(graph, node, n) for n in nn}
                new_edges = self.solve_intersection_with_constraint(graph, possible_edges, nodes_test_dict)

                if len(new_edges) > 0:
                    graph.remove_edges_from(list(graph.edges(node)))
                    graph.add_edges_from(new_edges)

        #############
        if int_bp:
            deg2_nodes = [n for n in graph.nodes if graph.degree(n) > 2]
            pairs = list(itertools.combinations(deg2_nodes, 2))
            possible_int_pairs = []
            for n1, n2 in pairs:
                if len(nx.shortest_path(graph, n1, n2)) < 4:
                    possible_int_pairs.append((n1, n2))

            for pair in possible_int_pairs:
                graph = self.solve_possible_int_pair(graph, pair)

        return graph

    def solve_possible_int_pair(self, graph, pair):
        n1, n2 = pair
        path = nx.shortest_path(graph, n1, n2)

        nodes_test1 = [n for n in list(graph.neighbors(n1)) if n not in path]
        nodes_test2 = [n for n in list(graph.neighbors(n2)) if n not in path]

        nodes_test1_dict = {}
        for n in nodes_test1:
            nn = [n for n in list(graph.neighbors(n)) if n not in path]
            if len(nn) > 0:
                nodes_test1_dict[n] = compute_edge_from_ids(graph, nn[0], n)

        nodes_test2_dict = {}
        for n in nodes_test2:
            nn = [n for n in list(graph.neighbors(n)) if n not in path]
            if len(nn) > 0:
                nodes_test2_dict[n] = compute_edge_from_ids(graph, nn[0], n)

        # return graph
        # compute best possible pair of edges for intersection
        permut = itertools.permutations(nodes_test1_dict.keys(), len(nodes_test2_dict.keys()))
        unique_combinations = [list(zip(comb, nodes_test2_dict.keys())) for comb in permut]
        nodes_test_combined = {**nodes_test1_dict, **nodes_test2_dict}
        possible_edges = [c for combs in unique_combinations for c in combs]

        # if len(path) == 2 -> solve intersection without constraints
        if len(path) < 3:
            check_constraint = False
        else:
            check_constraint = True
        new_edges = self.solve_intersection_with_constraint(
            graph, possible_edges, nodes_test_combined, check_constraint=check_constraint
        )

        if len(new_edges) > 0:
            graph.remove_edges_from(list(graph.edges(path)))
            graph.add_edges_from(new_edges)

        return graph

    def solve_intersection_with_constraint(self, graph, possible_edges, nodes_test_dict, check_constraint=True):
        # compute E
        E = {}
        combs_edges = list(itertools.combinations(possible_edges, 2))
        for v in combs_edges:
            for e0, e1 in v:
                dir_n0 = nodes_test_dict[e0]
                dir_n1 = nodes_test_dict[e1]
                dire = compute_edge_from_ids(graph, e0, e1)
                E[(e0, e1)] = cosine_sim_np(dir_n0, dire) * cosine_sim_np(dir_n1, dire)
        E = {k: v for k, v in sorted(E.items(), key=lambda item: item[1])}

        print(E)

        ################
        # compute new edges
        return self.compute_new_edges_with_constraints(E, check_constraint)

    def compute_new_edges_with_constraints(self, edge_score, check_constraint=True):
        counter = 0
        new_edges, nodes_done = [], []
        for ke, _ in edge_score.items():
            if counter > 2:
                break

            if ke[0] not in nodes_done and ke[1] not in nodes_done:
                new_edges.append(ke)
                nodes_done.extend([ke[0], ke[1]])
                counter += 1

        if check_constraint:
            for edge in new_edges:
                print("edge", edge, edge_score[edge])
                if edge_score[edge] > -0.5:
                    return []

        return new_edges
