import torch
from torch import nn
from torch_geometric import nn as pyg_nn
from gnn_dmlo.model.layers import GeneralConv


class LinkPredictorDecoder(torch.nn.Module):
    def __init__(self, config, num_layers=3, out_dim=1):
        super(LinkPredictorDecoder, self).__init__()

        mid_dim = config["hidden_dim"]

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(mid_dim, mid_dim))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(mid_dim, mid_dim))
        self.lins.append(torch.nn.Linear(mid_dim, out_dim))

    def forward(self, x, edge_index, sigmoid=True):
        z = x[edge_index[0]] * x[edge_index[1]]

        for lin in self.lins[:-1]:
            z = lin(z).relu()
        z = self.lins[-1](z)
        if sigmoid:
            return torch.sigmoid(z)
        else:
            return z


class ConvNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(1, hidden_dim // 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(hidden_dim // 2)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        y = self.conv1(x).relu()
        y = self.bn1(y)
        y = self.conv2(y).relu()
        y = self.bn2(y)
        y = self.max_pool(y)
        return y


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.get("hidden_dim", 32)
        self.edge_norm_enc = config.get("edge_norm_enc", False)
        self.edge_dir_enc = config.get("edge_dir_enc", False)

        self.edge_enc = self.edge_dir_enc or self.edge_norm_enc
        # print("GNN Encoder | edge features: ", self.edge_enc)

        act = config.get("activation", "relu")  # relu, prelu
        if act == "relu":
            self.activation = nn.ReLU()
        elif act == "prelu":
            self.activation = nn.PReLU()
        else:
            raise ValueError("activation function should be 'relu' or 'prelu'")

        # NODE EMBEDDINGS
        self.embedding_coord = nn.Linear(2, self.hidden_dim)
        self.emb_norm = nn.LayerNorm(self.hidden_dim * 2)
        self.f_node = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # EDGE EMBEDDINGS
        counter = 0
        self.edge_dim = None
        if self.edge_norm_enc:
            self.embedding_edge_norm = nn.Linear(1, self.hidden_dim // 2)
            counter += 1

        if self.edge_dir_enc:
            self.embedding_edge_dir = nn.Linear(180, self.hidden_dim // 2)
            counter += 1

        if self.edge_enc:
            self.f_edge = nn.Linear(counter * self.hidden_dim // 2, self.hidden_dim)
            self.edge_dim = self.hidden_dim

        self.conv1 = GeneralConv(self.hidden_dim, self.hidden_dim, in_edge_channels=self.edge_dim)
        self.conv2 = GeneralConv(self.hidden_dim, self.hidden_dim, in_edge_channels=self.edge_dim)
        self.conv3 = GeneralConv(self.hidden_dim, self.hidden_dim, in_edge_channels=self.edge_dim)

        self.bn1 = pyg_nn.BatchNorm(self.hidden_dim)
        self.bn2 = pyg_nn.BatchNorm(self.hidden_dim)

    def encoding_edge(self, data):
        return self.embedding_edge_norm(data.edge_norm).relu()

    def forward(self, data, zm):
        zpos = self.embedding_coord(data.x).relu()
        z = self.emb_norm(torch.cat([zpos, zm], axis=-1))
        z = self.f_node(z).relu()

        if not self.edge_enc:
            z = self.bn1(self.conv1(z, data.edge_index)).relu()
            z = self.bn2(self.conv2(z, data.edge_index)).relu()
            z = self.conv3(z, data.edge_index)
            return z

        # EDGES
        ze = []
        if self.edge_norm_enc:
            zenorm = self.activation(self.embedding_edge_norm(data.edge_norm))
            ze.append(zenorm)

        if self.edge_dir_enc:
            zedir = self.activation(self.embedding_edge_dir(data.edge_dir_smooth))
            ze.append(zedir)

        if len(ze) == 1:
            ze = ze[0]
        else:
            ze = torch.cat(ze, dim=1)

        ze = self.activation(self.f_edge(ze))

        z = self.bn1(self.conv1(z, data.edge_index, ze))
        z = self.activation(z)
        z = self.bn2(self.conv2(z, data.edge_index, ze))
        z = self.activation(z)
        z = self.conv3(z, data.edge_index, ze)
        return z


class NodeAngleGNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.activation = nn.ReLU()
        self.edge_w_enc = nn.Linear(1, hidden_dim // 2)
        self.conv1 = GeneralConv(hidden_dim, hidden_dim, in_edge_channels=hidden_dim // 2)
        self.lin_angle_out = nn.Linear(hidden_dim, 180)

    def forward(self, data, zm, zw):
        ze = self.activation(self.edge_w_enc(zw))

        z = self.activation(self.conv1(zm, data.edge_index, ze))
        return self.lin_angle_out(z)


class Network(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = pyg_nn.GAE(encoder=Encoder(config), decoder=LinkPredictorDecoder(config))

        model_parameters_number = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of parameters: ", model_parameters_number)

        self.convnet_encoder = ConvNet(config["hidden_dim"])

        model_parameters_number = sum(p.numel() for p in self.convnet_encoder.parameters() if p.requires_grad)
        print("Number of parameters: ", model_parameters_number)

        self.angle_gnn = NodeAngleGNN(config["hidden_dim"])

        model_parameters_number = sum(p.numel() for p in self.angle_gnn.parameters() if p.requires_grad)
        print("Number of parameters: ", model_parameters_number)

        self.criterion = nn.BCEWithLogitsLoss()

    def encode_mask(self, data):
        z_mask = self.convnet_encoder(data.mask).relu()

        if z_mask.shape[0] == 1:
            z_tmp = torch.index_select(z_mask, -2, data.rows.view(-1))
            z_tmp = torch.index_select(z_tmp, -1, data.cols.view(-1))
            zm = torch.diagonal(z_tmp, dim1=-2, dim2=-1).squeeze().permute(1, 0)
            return zm
        else:
            zm = []
            for b in range(z_mask.shape[0]):
                z_tmp = torch.index_select(z_mask[b], -2, data.rows[data.batch == b].view(-1))
                z_tmp = torch.index_select(z_tmp, -1, data.cols[data.batch == b].view(-1))
                zm.append(torch.diagonal(z_tmp, dim1=-2, dim2=-1).squeeze().permute(1, 0))

            return torch.cat(zm, axis=0)

    def forward(self, data):
        zm = self.encode_mask(data)

        z = self.model.encode(data, zm)
        zw = self.model.decode(z, data.edge_index)

        zm_angle = self.angle_gnn(data, zm, zw)

        return {"node_feat": z, "node_angle": zm_angle, "mask_enc": zm, "edge_weight": zw}

    def losses(self, data, z, z_angle):
        loss_ae = self.model.recon_loss(z, data.edge_pos.T, data.edge_neg.T)
        loss_angle = self.criterion(z_angle, data.node_angle_gauss)
        return loss_ae, loss_angle

    def test(self, data, z):
        auc, ap = self.model.test(z, data.edge_pos.T, data.edge_neg.T)
        return auc, ap

    def predict(self, data):
        res = self.forward(data)
        z = res["node_feat"]
        zm = res["mask_enc"]
        pred_weight = res["edge_weight"]
        pred_angle = res["node_angle"].softmax(dim=-1)
        pred_angle = pred_angle / torch.max(pred_angle)
        return {"pred_weight": pred_weight, "pred_angle": pred_angle, "node_feat": z, "mask_enc": zm}


#####################################


class SubGraphClassificator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.get("hidden_dim", 32)
        self.out_dim = 1
        self.enc_nodes_dim = config.get("enc_nodes_dim", 32)
        self.mask_enc = config.get("mask_enc", False)
        self.lp_fea_enc = config.get("lp_fea_enc", False)
        self.edge_dir_enc = config.get("edge_dir_enc", False)
        self.edge_norm_enc = config.get("edge_norm_enc", False)
        self.num_layers = config.get("gnn_layers", 1)

        # ENCODING NODES
        self.embedding_pos = nn.Linear(2, self.hidden_dim // 2)
        counter_node_enc = 1
        if self.mask_enc:
            self.embedding_mask = nn.Linear(self.enc_nodes_dim, self.hidden_dim // 2)
            counter_node_enc += 1

        if self.lp_fea_enc:
            self.embedding_fea = nn.Linear(self.enc_nodes_dim, self.hidden_dim // 2)
            counter_node_enc += 1

        self.f_node = nn.Linear(counter_node_enc * self.hidden_dim // 2, self.hidden_dim)
        self.f_node_norm = nn.LayerNorm(self.hidden_dim)

        # ENCODING EDGES
        counter_edge_enc = 0
        if self.edge_norm_enc:
            self.embedding_edge_norm = nn.Linear(1, self.hidden_dim // 2)
            counter_edge_enc += 1

        if self.edge_dir_enc:
            self.embedding_edge_dir = nn.Linear(180, self.hidden_dim // 2)
            counter_edge_enc += 1

        if counter_edge_enc > 0:
            self.f_edge = nn.Linear(counter_edge_enc * self.hidden_dim // 2, self.hidden_dim)

        # GRAPH CONVOLUTIONS
        self.layers = nn.ModuleList([])
        for _ in range(self.num_layers):
            conv = GeneralConv(self.hidden_dim, self.hidden_dim, in_edge_channels=self.hidden_dim)
            self.layers.append(conv)

        # OUTPUT CLS
        self.lin = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.lin_out = nn.Linear(self.hidden_dim // 2, self.out_dim)

    def forward(self, data):
        # NODES
        zpos = self.embedding_pos(data.x).relu()

        znode = [zpos]
        if self.mask_enc:
            zm = self.embedding_mask(data.mask_enc).relu()
            znode.append(zm)

        if self.lp_fea_enc:
            zfea = self.embedding_fea(data.z).relu()
            znode.append(zfea)

        z = self.f_node(torch.cat(znode, axis=-1)).relu()
        z = self.f_node_norm(z)

        # EDGES
        ze = []
        if self.edge_norm_enc:
            zenorm = self.embedding_edge_norm(data.edge_norm).relu()
            ze.append(zenorm)

        if self.edge_dir_enc:
            zedir = self.embedding_edge_dir(data.edge_dir_smooth).relu()
            ze.append(zedir)

        if ze:
            ze = torch.cat(ze, dim=1) if len(ze) > 1 else ze[0]
            ze = self.f_edge(ze).relu()
        else:
            ze = None

        # AGGREGATE and UPDATE
        for layer in self.layers:
            if ze is not None:
                z = layer(z, data.edge_index, ze).relu()
            else:
                z = layer(z, data.edge_index).relu()

        z = pyg_nn.global_max_pool(z, data.batch)
        z = self.lin(z).relu()
        return self.lin_out(z)
