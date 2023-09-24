import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pickle
from ptq2theMoon.ptqTransformerBased import PTQForTransformerBased
import numpy as np
import torch.nn as nn

def visualize_attention(img, patch_size, attentions):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, \
        img.shape[2] -  img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    # attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1]  # number of head

    # keep only the output patch attention
    process_attentions = []
    for i in range(attentions.shape[0]):
        process_attn = attentions[i, :, 0, 1:].reshape(nh, -1)
        process_attn = process_attn.reshape(nh, w_featmap, h_featmap)
        process_attn = nn.functional.interpolate(process_attn.unsqueeze(
            0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()
        process_attentions.append(process_attn)

    return process_attentions


def plot_attention(img, attention):
    # attention = attention.cpu().numpy()
    n_heads = attention.shape[0]
    print(f'plotting attention: {attention.shape}, {n_heads}')

    # img = mpimg.imread('your_image.png')
    # heatmap = np.random.rand(224, 224)
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.imshow(heatmap, cmap='jet', alpha=0.5)
    # plt.show()

    plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean", "Heat Image"]
    h_img = img * np.tile(np.expand_dims(np.mean(attention, 0), axis=-1), (1, 1, 3))
    for i, fig in enumerate([img, np.mean(attention, 0), img]):
        if i == 2:
            plt.imshow(fig, cmap=plt.cm.gray)
            plt.imshow(np.mean(attention, 0), cmap='inferno', alpha=0.5)
            plt.title(text[i])
            break
        # plt.subplot(1, 3, i+1)
        # plt.imshow(fig, cmap='inferno')
        # plt.title(text[i])
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(n_heads):
        plt.subplot(n_heads//3, 3, i+1)
        plt.imshow(attention[i], cmap='inferno')
        plt.title(f"Head n: {i+1}")
    plt.tight_layout()
    plt.show()

