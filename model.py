import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelPredictor(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels_cnn=8,
                 fc_hidden_dims=[64, 32],
                 voxel_max_dims=(25, 25, 50),
                 meta_info_dim=2):
        super(VoxelPredictor, self).__init__()

        self.input_channels = input_channels
        self.voxel_max_dims = voxel_max_dims
        self.meta_info_dim = meta_info_dim

        self.final_cnn_channels = output_channels_cnn * 4

        self.cnn1_feat_extractor = nn.Sequential(
            nn.Conv3d(input_channels, output_channels_cnn, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels_cnn),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(output_channels_cnn, output_channels_cnn * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels_cnn * 2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(output_channels_cnn * 2, self.final_cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.final_cnn_channels),
            nn.ReLU(),
        )

        self.cnn2_feat_extractor = nn.Sequential(
            nn.Conv3d(input_channels, output_channels_cnn, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels_cnn),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(output_channels_cnn, output_channels_cnn * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels_cnn * 2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(output_channels_cnn * 2, self.final_cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.final_cnn_channels),
            nn.ReLU(),
        )

        dummy_input = torch.rand(1, input_channels,
                                 voxel_max_dims[2],
                                 voxel_max_dims[1],
                                 voxel_max_dims[0]) 

        with torch.no_grad(): 
            dummy_output = self.cnn1_feat_extractor(dummy_input)
            cnn_output_flat_dim = dummy_output.view(dummy_output.size(0), -1).size(1)

        self.fc_input_dim = (cnn_output_flat_dim * 2) + self.meta_info_dim

        fc_layers = []
        in_dim = self.fc_input_dim
        for h_dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(in_dim, 128))
            fc_layers.append(nn.Linear(128, 128))
            fc_layers.append(nn.Linear(128, 128))
            fc_layers.append(nn.Linear(128, h_dim))
            fc_layers.append(nn.ReLU())
            in_dim = h_dim

        fc_layers.append(nn.Linear(in_dim, 1))

        self.fc_layers = nn.Sequential(*fc_layers)

    def _calculate_output_dim(self, input_dim, num_pools):
        output_dim = input_dim
        for _ in range(num_pools):
            output_dim = (output_dim - 2) // 2 + 1
        return output_dim

    def _apply_cnn_attention(self, feature_map):
        channel_att = self.channel_attention(feature_map)
        feature_map = feature_map * channel_att 

        avg_out = torch.mean(feature_map, dim=1, keepdim=True)
        max_out = torch.max(feature_map, dim=1, keepdim=True)[0] 
        spatial_input = torch.cat([avg_out, max_out], dim=1) 

        spatial_att = self.spatial_attention(spatial_input)
        feature_map = feature_map * spatial_att 

        return feature_map, channel_att, spatial_att 

    def forward(self, voxel1, voxel2, meta_info):

        feat1_raw = self.cnn1_feat_extractor(voxel1)
        feat2_raw = self.cnn2_feat_extractor(voxel2)
        feat1_flattened = feat1_raw.view(feat1_raw.size(0), -1)
        feat2_flattened = feat2_raw.view(feat2_raw.size(0), -1)

        combined_features_cnn = torch.cat((feat1_flattened, feat2_flattened), dim=1)

        if meta_info.dtype != torch.float32:
            meta_info = meta_info.float()

        combined_features_final = torch.cat((combined_features_cnn, meta_info), dim=1)

        output = self.fc_layers(combined_features_final)
        return output
