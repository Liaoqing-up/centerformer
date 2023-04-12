import os
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np


def get_common_sample_token_from_two_dir(dir_path1, dir_path2):
    files1 = os.listdir(dir_path1)
    files2 = os.listdir(dir_path2)
    common_files = list(set(files1).intersection(set(files2)))
    return common_files

def plot_heatmaps_from_two_dir(dir_path1, dir_path2, save=False):
    common_files = get_common_sample_token_from_two_dir(dir_path1, dir_path2)
    for file in common_files:
        path1 = os.path.join(dir_path1, file)
        path2 = os.path.join(dir_path2, file)
        img1 = mpimg.imread(path1)
        img2 = mpimg.imread(path2)
        titles = [path.split('/')[-1] for path in [dir_path1, dir_path2]]
        plt.figure()
        for i, img in enumerate([img1, img2]):
            ax = plt.subplot(1,2,i+1)
            ax.set_title(titles[i])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.axis('off')
        if save:
            plt.savefig(
                f'/opt/sdatmp/lq/project/centerformer/debug/heatmaps/common_red/{file}.jpg',
                bbox_inches = 'tight',
                dpi = 1200
            )
        plt.close()
            

if __name__ == "__main__":
    dir_path1 = '/opt/sdatmp/lq/project/centerformer/debug/heatmaps/no_tsa/'
    dir_path2 = '/opt/sdatmp/lq/project/centerformer/debug/heatmaps/tsa/'
    plot_heatmaps_from_two_dir(dir_path1, dir_path2, save=True)