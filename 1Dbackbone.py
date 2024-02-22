import torch
import torch.nn as nn

class TimeSeriesCNNBackbone(nn.Module):
    def __init__(self, input_height, input_width, num_filters=(32, 64, 128), kernel_sizes=(3, 3, 3), pool_sizes=(2, 2, 2)):
        super(TimeSeriesCNNBackbone, self).__init__()
        layers = []
        in_channels = 1  # 1D CNN chỉ chấp nhận dữ liệu có 1 kênh (chiều) đầu vào
        for out_channels, kernel_size, pool_size in zip(num_filters, kernel_sizes, pool_sizes):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_size))
            layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        # Tính toán kích thước output của Conv1d để sử dụng trong phần Linear của mạng
        conv_output_height = self.calculate_conv_output_height(input_height, kernel_sizes, pool_sizes)
        conv_output_width = input_width // max(pool_sizes)  # Giả sử sử dụng kernel_size và pool_size lớn nhất
        self.fc_input_dim = conv_output_height * num_filters[-1] * conv_output_width
        self.fc = nn.Linear(self.fc_input_dim, 512)  # Fully connected layer
    
    def forward(self, x):
        # Số lượng kênh (channels) trong dữ liệu input được mở rộng để phù hợp với kích thước của Conv1d
        x = x.unsqueeze(1)  # Thêm một kích thước chiều vào dữ liệu đầu vào để phù hợp với Conv1d
        x = self.features(x)
        # x = x.view(-1, self.fc_input_dim)  # Flatten output của Conv1d
        # x = self.fc(x)
        return x
    
    def calculate_conv_output_height(self, input_height, kernel_sizes, pool_sizes):
        output_height = input_height
        for kernel_size, pool_size in zip(kernel_sizes, pool_sizes):
            output_height = (output_height - kernel_size + 1) // pool_size
        return output_height

# Sử dụng backbone
input_height = 60
input_width = 60
backbone = TimeSeriesCNNBackbone(input_height, input_width)

# Test với một tensor giả
x = torch.randn(input_height, input_width)  # input_length là độ dài của chuỗi thời gian
output = backbone(x)
print("Output shape:", output.shape)
