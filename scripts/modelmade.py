import torch
from torchvision import models

# 定义模型
#model = models.vgg16(pretrained=True)
model = models.resnet50(pretrained=True)
# 定义输入
input = torch.randn(1, 3, 224, 224)

# 跟踪模型
traced_model = torch.jit.trace(model, input)

# 保存模型
#traced_model.save("vgg16.pt")
traced_model.save("resnet50.pt")
# # 加载模型
# model = torch.jit.load("vgg16.pt")

# # 将模型移动到 GPU
# model.cuda()

# # 运行模型
# output = model(input.cuda())

# # 打印输出
# print(output)