import time
import torch
from dataloader import ClsDataset
from torch.utils.data import DataLoader
import albumentations as A
from graphs.swin_transformer import SwinTransformer
from graphs.efficientnet_pytorch import EfficientNet
"""
SwinTransformer和efficientnet-b0耗时基本相同没有差异，但是精度要更高。
"""

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model_name = "efficientnet"
    size = 224

    if model_name == "swin_transformer":
        model = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False).to(device)
        model.load_state_dict(torch.load("assets/weights/swin_20_98.333.pt", map_location=device))
    elif model_name == "efficientnet":
        model = EfficientNet.from_name("efficientnet-b0", in_channels=3, num_classes=3).to(device)
        model.load_state_dict(torch.load("assets/weights/efficientnet-b0_30_95.0.pt", map_location=device))
    else:
        model = None
        exit()
    model.eval()

    trans = A.Compose([
        A.Resize(height=size, width=size, p=1),
        A.Normalize(p=1, mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616])
    ])

    dataset = ClsDataset("data/images", "data/val.txt", transform=trans)
    data_loader = DataLoader(dataset, batch_size=1)
    print("%d data loaded" % len(dataset))

    _ = model(torch.rand(1, 3, size, size).cuda())  # 第一次inference时间会非常长

    A, B = [], []
    for i in range(5):  # 数据太少，多循环几次
        START = time.time()
        inference_times = []
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                # 这里有个大坑，注意注意注意
                torch.cuda.synchronize()
                start = time.time()
                outputs = model(images)
                torch.cuda.synchronize()
                inference_times.append(time.time() - start)
        END = time.time()
        A.append(sum(inference_times)/len(inference_times))
        B.append((END-START) / len(dataset))
    print(f"{model_name} 纯推理时间:{sum(A) / len(A) * 1000}ms")  # 10ms（GTX2060)
    print(f"{model_name} 推理加数据处理时间:{sum(B) / len(B) * 1000}ms")  # 16ms
