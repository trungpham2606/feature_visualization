import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import torch.nn as nn
import cv2


def visualize(model, image):
    #preprocess image
    if not torch.is_tensor(image):
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(np.array(image))
        image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)

    #feeding to network
    counter = 0
    model_list = list(model.children())
    model_weights = []
    convs_layers = []

    for i in range(len(model_list)):
        if type(model_list[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_list[i].weight)
            convs_layers.append(model_list[i])
        elif type(model_list[i]) == nn.Sequential:
            for j in range(len(model_list[i])):
                for child in model_list[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        convs_layers.append(child)
    print('Total convolutional layers: {}'.format(counter))

    outputs = [convs_layers[0](image)]
    for i in range(1, len(convs_layers)):
        outputs.append(convs_layers[i](outputs[-1]))
    
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64: # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        plt.show()


if __name__ == '__main__':
    image = cv2.imread(r'F:\data\FSC147_384_V2\test\7521.jpg')
    model = resnet18(pretrained=True)
    visualize(model, image)
    

