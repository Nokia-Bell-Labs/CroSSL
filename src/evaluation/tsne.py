import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from ds_utils.data_helper import load_dataset, get_ds_info
#import imageio

class EmbeddingVisualisation:
    def __init__(self, plot_dir, class_size):
        self.class_size = class_size
        self.plot_dir = plot_dir
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)

    def plot_tsne(self, data, lbl, exp_id, label_name, tb_fw = None):
        target = np.array(np.argmax(lbl, axis=-1))
        tsne = TSNE(2, learning_rate='auto', verbose=0)
        tsne_proj = tsne.fit_transform(data)
        # Plot those points as a scatter plot and label them based on the pred labels
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(8, 8))
        num_categories = self.class_size
        for lab in range(num_categories):
            indices = target == lab
            ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(lab)).reshape(1, 4),
                       label=lab, alpha=0.5)
        # ax.axis('off')
        plt.title(exp_id)
        ax.legend(label_name, fontsize='large', markerscale=2)


        plt.savefig(os.path.join(self.plot_dir, exp_id + ".png"))
        print('saved t-SNE at ', os.path.join(self.plot_dir, exp_id + ".png"))
        # Using the file writer, log the reshaped image.
        #img = imageio.imread(os.path.join(self.plot_dir, exp_id + ".png"))
        #with tb_fw.as_default():
        #    tf.summary.image("T-SNE", img, step=0)

def tsne_visualize(data_path, ds_name, working_dir, base_encoder, exp_id, mode="fixed", raw_data=False, tb_fw=None):
    print("t-SNE visualization of learnt representation:")
    ds_info = get_ds_info(ds_name)
    plots_path = os.path.join(working_dir,"plots")
    vis = EmbeddingVisualisation(plots_path, class_size=ds_info['class_size'])
    data = load_dataset(data_path, ds_name, state="all", label_efficiency=1)

    lbl = list(tf.concat([y for x, y in data], axis=0))

    if raw_data:
        vis.plot_tsne(np.array(tf.concat([tf.concat(item, axis=-1) for item in [x for x,y in data]], axis=0)), lbl, "TSNE_" + exp_id + "raw", ds_info['labels'])

    embeddings = np.array(tf.concat([tf.concat(item, axis=-1) for item in [base_encoder(x) for x, y in data]], axis=0))
    vis.plot_tsne(embeddings, lbl, "TSNE_" + exp_id + mode, ds_info['labels'], tb_fw= tb_fw)
