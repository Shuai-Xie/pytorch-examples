import torch
from torchvision.models import resnet18
# from torch.utils.tensorboard import SummaryWriter  # torch 自带的只能观测参数
from tensorboardX import SummaryWriter  # tensorboardX 才能查看模型结构


def tb_vis():
    x = torch.rand((1, 3, 224, 224))
    model = resnet18()
    writer = SummaryWriter(comment='_res18')  # CURRENT_DATETIME_HOSTNAME_res18
    writer.add_graph(model, input_to_model=x)


if __name__ == '__main__':
    tb_vis()