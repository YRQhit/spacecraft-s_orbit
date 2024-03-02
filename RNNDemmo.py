#
# import torch
# import torch.nn as nn
#
# class CustomRNNActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, has_continuous_action_space):
#         super(CustomRNNActorCritic, self).__init__()
#
#         self.has_continuous_action_space = has_continuous_action_space
#         self.hidden_size = 64  # 你可以根据需要调整隐藏层大小
#
#         # actor
#         if has_continuous_action_space:
#             self.actor = nn.Sequential(
#                 nn.Linear(state_dim, self.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(self.hidden_size, self.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(self.hidden_size, action_dim),
#                 nn.Tanh()
#             )
#         else:
#             self.actor = nn.Sequential(
#                 nn.Linear(state_dim, self.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(self.hidden_size, self.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(self.hidden_size, action_dim),
#                 nn.Softmax(dim=-1)
#             )
#
#         # critic (RNN)
#         self.rnn = nn.RNN(input_size=state_dim, hidden_size=self.hidden_size, batch_first=True)
#         self.critic = nn.Sequential(
#             nn.Linear(self.hidden_size, 1)
#         )
#
#     def forward(self, x):
#         # 使用RNN进行特征提取
#         rnn_out, _ = self.rnn(x.unsqueeze(0))
#
#         # 获取RNN的最后一个时间步的输出
#         # rnn_out = rnn_out[:, -1, :]
#
#         # actor部分
#
#
#         # critic部分
#         critic_out = self.critic(rnn_out)
#
#         return  critic_out
#
# # 创建自定义RNN Actor-Critic模型实例
# custom_rnn_model = CustomRNNActorCritic(state_dim=6, action_dim=3, has_continuous_action_space=True)
#
# # 创建一个随机输入序列，假设序列长度是5，每个时间步有10个特征
# input_sequence = torch.randn(6)
#
# # 将输入传递给模型
# critic_output = custom_rnn_model(input_sequence)
# print("Critic Output:", critic_output)

import torch
import torch.nn as nn

class CustomRNNActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, num_rnn_layers=1):
        super(CustomRNNActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.hidden_size = 64  # 你可以根据需要调整隐藏层大小
        self.num_rnn_layers = num_rnn_layers

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic (RNN)
        self.rnn = nn.RNN(input_size=state_dim, hidden_size=self.hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, x):
        # 使用多层RNN进行特征提取
        rnn_out, _ = self.rnn(x.unsqueeze(0).unsqueeze(0))

        # 获取RNN的最后一个时间步的输出
        rnn_out = rnn_out[:, -1, :]

        # actor部分

        # critic部分
        critic_out = self.critic(rnn_out)

        return critic_out

# 创建自定义多层RNN Actor-Critic模型实例，设置num_rnn_layers参数为2
custom_rnn_model = CustomRNNActorCritic(state_dim=6, action_dim=3, has_continuous_action_space=True, num_rnn_layers=2)

# 创建一个随机输入序列，假设序列长度是5，每个时间步有6个特征
input_sequence = torch.randn(10,6)

# 将输入传递给模型
critic_output = custom_rnn_model(input_sequence)
print("Critic Output:", critic_output)
