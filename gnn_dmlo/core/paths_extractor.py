import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class PathsProc:
    def __init__(self):
        self.debug = False

    def run(self, graph, mask_img):

        paths = self.extract_paths(graph)

        if self.debug:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(mask_img, cmap="gray")
            for k, v in paths.items():
                points = v["points"]
                axs[0].plot(points[:, 1], points[:, 0], "o-", label=f"path {k}")

            axs[0].legend()
            axs[0].axis("equal")
            axs[0].set_title("paths")

            for n in graph.nodes:
                pos = graph.nodes[n]["pos"]
                axs[1].plot(pos[1], pos[0], "o", color="tab:blue")
            for e in graph.edges:
                pos1 = graph.nodes[e[0]]["pos"]
                pos2 = graph.nodes[e[1]]["pos"]
                axs[1].plot([pos1[1], pos2[1]], [pos1[0], pos2[0]], "-", color="black")

            axs[1].axis("equal")
            axs[1].set_title("graph")

            plt.tight_layout()
            plt.show()

        return paths, graph

    def check_paths_consistency(self, graph, paths):
        def cosine_sim_np(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        if len(paths) == 0:
            return {}

        curr_key = max(list(paths.keys())) + 1
        new_paths = {}
        edges_to_remove = []
        for k, path in paths.items():
            points_divide = []
            nodes = path["nodes"]
            points = path["points"]

            if len(nodes) < 3:
                new_paths[k] = {"points": points, "nodes": nodes}
                continue

            nodes_pos_1 = np.array([graph.nodes[nodes[it - 1]]["pos"] for it in range(1, len(nodes) - 1)]).astype(
                float
            )
            nodes_pos_2 = np.array([graph.nodes[nodes[it]]["pos"] for it in range(1, len(nodes) - 1)]).astype(float)
            nodes_pos_3 = np.array([graph.nodes[nodes[it + 1]]["pos"] for it in range(1, len(nodes) - 1)]).astype(
                float
            )

            dirs12 = nodes_pos_2 - nodes_pos_1
            dirs12_norms = np.linalg.norm(dirs12, axis=1)
            dirs12[:, 0] = dirs12[:, 0] / dirs12_norms
            dirs12[:, 1] = dirs12[:, 1] / dirs12_norms

            dirs23 = nodes_pos_3 - nodes_pos_2
            dirs23_norms = np.linalg.norm(dirs23, axis=1)
            dirs23[:, 0] = dirs23[:, 0] / dirs23_norms
            dirs23[:, 1] = dirs23[:, 1] / dirs23_norms

            dirs = np.array([graph.nodes[nodes[it + 1]]["angle"] for it in range(1, len(nodes) - 1)])
            dirs_norms = np.linalg.norm(dirs, axis=1)
            dirs[:, 0] = dirs[:, 0] / dirs_norms
            dirs[:, 1] = dirs[:, 1] / dirs_norms

            for it in range(1, len(nodes) - 1):
                s12 = cosine_sim_np(dirs[it - 1, :], dirs12[it - 1, :])
                s23 = cosine_sim_np(dirs[it - 1, :], dirs23[it - 1, :])

                if s12 * s23 < 0:
                    points_divide.append(it)

            if not points_divide:
                new_paths[k] = {"points": points, "nodes": nodes}
            else:
                for p in points_divide:
                    seq0 = nodes[:p]
                    points0 = points[:p]
                    if len(points0) > 2:
                        new_paths[curr_key] = {"points": points0, "nodes": seq0}
                        curr_key += 1

                    points1 = points[p + 1 :]
                    seq1 = nodes[p + 1 :]
                    if len(points1) > 2:
                        new_paths[curr_key] = {"points": points1, "nodes": seq1}
                        curr_key += 1

                    edges_to_remove.append((nodes[p], nodes[p + 1]))

        print("consistency edges to remove:", edges_to_remove)
        # graph.remove_edges_from(edges_to_remove)
        return new_paths, graph

    def extract_paths(self, G):
        # remove edges from nodes with degree > 2 to split the graph in connected components
        Gtmp = G.copy()
        division_nodes = [n for n in G.nodes if G.degree(n) > 2]
        Gtmp.remove_edges_from(G.edges(division_nodes))

        paths_dict = {}
        for it, nodes in enumerate(list(nx.connected_components(Gtmp))):
            endpoints = [n for n in nodes if Gtmp.degree(n) == 1]
            path_nodes = None

            if len(endpoints) == 2:  # normal case
                path_nodes = nx.shortest_path(Gtmp, endpoints[1], endpoints[0])

                neigh_0 = [n for n in G.neighbors(endpoints[0]) if n not in path_nodes]
                if len(neigh_0) == 1:
                    path_nodes.append(neigh_0[0])

                neigh_1 = [n for n in G.neighbors(endpoints[1]) if n not in path_nodes]
                if len(neigh_1) == 1:
                    path_nodes.insert(0, neigh_1[0])

            elif len(endpoints) == 0:  # single point between branch points
                neighs = [n for n in G.neighbors(list(nodes)[0])]
                if len(neighs) == 2:
                    path_nodes = [neighs[0], list(nodes)[0], neighs[1]]

            # add to output
            if path_nodes is not None:
                points = np.array([G.nodes[n]["pos"] for n in path_nodes])
                paths_dict[it] = {"nodes": path_nodes, "points": points}

        return paths_dict

    def merge_paths(self, paths, paths_to_merge):
        # update paths
        paths_ids_updated = {}
        nodes_done = []
        for data in paths_to_merge:
            key1 = data["path1"]
            key2 = data["path2"]
            node1 = data["node1"]
            node2 = data["node2"]

            if node1 in nodes_done or node2 in nodes_done:
                continue

            if key1 in paths_ids_updated:
                key1 = paths_ids_updated[key1]

            if key2 in paths_ids_updated:
                key2 = paths_ids_updated[key2]

            nodes1 = paths[key1]["nodes"]
            nodes2 = paths[key2]["nodes"]
            points1 = paths[key1]["points"]
            points2 = paths[key2]["points"]

            if node1 == nodes1[0]:
                nodes1 = nodes1[::-1]
                points1 = points1[::-1]

            if node2 == nodes2[-1]:
                nodes2 = nodes2[::-1]
                points2 = points2[::-1]

            nodes = np.concatenate([nodes1, nodes2])
            points = np.concatenate([points1, points2])

            paths[key1] = {"nodes": nodes, "points": points}
            paths_ids_updated[key2] = key1
            nodes_done.extend([node1, node2])
            del paths[key2]

        return paths


############################################################################################################
