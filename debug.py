# import argparse
#
# def str2bool(v):
#     return v.lower() in ("yes", "true", "t", "1")
#
# VOC_ROOT = r"./"
#
# parser = argparse.ArgumentParser(
#     description='Single Shot MultiBox Detector Evaluation')
# parser.add_argument('--trained_model',
# #                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
#                     default='weights/ssd300_COCO_45000.pth', type=str,
#                     help='Trained state_dict file path to open')
# parser.add_argument('--save_folder', default='eval/', type=str,
#                     help='File path to save results')
# parser.add_argument('--confidence_threshold', default=0.01, type=float,
#                     help='Detection confidence threshold')
# parser.add_argument('--top_k', default=5, type=int,
#                     help='Further restrict the number of predictions to parse')
# parser.add_argument('--cuda', default=True, type=str2bool,
#                     help='Use cuda to train model')
# parser.add_argument('--voc_root', default=VOC_ROOT,
#                     help='Location of VOC root directory')
# parser.add_argument('--cleanup', default=True, type=str2bool,
#                     help='Cleanup and remove results files following eval')
# parser.add_argument('--annotation_folder', default='./', type=str)
# parser.add_argument('--image_folder', default='./', type=str)
# parser.add_argument('--image_set_file', default='./', type=str)
#
# args = parser.parse_args()
#
# print(args)
# print(args.trained_model)
# print(args.save_folder)
# print(args.cuda)
#
# import torch
# print(torch.cuda.is_available())
#
#
#
#
import torch
import torchvision
import torchvision.models as models
import time
import numpy as np


def test_on_device(model, dump_inputs, warn_up, loops, device_type):
    if device_type == 'cuda':
        assert torch.cuda.is_available()
    device = torch.device(device_type)

    # model = models.alexnet.alexnet(pretrained=False).to(device)
    model.to(device)
    model.eval()
    dump_inputs = dump_inputs.to(device)

    with torch.no_grad():
        executions = []
        for i in range(warn_up + loops):
            if device_type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = model(dump_inputs)
            if device_type == 'cuda':
                torch.cuda.synchronize()  # CUDA sync
            end = time.time()
            executions.append((end - start) * 1000)  # ms
    # print(f'Avg time:{np.mean(executions)} ms')
    return np.mean(executions[warn_up:])


if __name__ == "__main__":
    print(torch.cuda.is_available())
    model_list = {
        'AlexNet': models.alexnet(),
        'ResNet-50': models.resnet50(),
        'ResNet-18': models.resnet18(),
        'ResNet-101': models.resnet101(),
        'MobileNet-v2': models.mobilenet_v2(),
        'SqueezeNet1-1': models.squeezenet1_1()
    }

    batch_size = 1
    for name, model in model_list.items():
        print('=' * 10 + f'{name}' + '=' * 10)
        avg_time = test_on_device(model=model, dump_inputs=torch.rand(batch_size, 3, 224, 224), warn_up=3, loops=10,
                                  device_type='cuda')
        print(f'Avg time:{avg_time} ms')





