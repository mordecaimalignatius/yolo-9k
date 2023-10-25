"""
custom version of original ultralytics yolov8 ConfusionMatrix

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from ultralytics.utils import LOGGER, plt_settings, TryExcept
from ultralytics.utils.metrics import ConfusionMatrix


class ConfusionMatrix9000(ConfusionMatrix):
    def metrics(self, names: () = None):
        """ return additional metrics """
        total = self.matrix.sum(axis=0)
        total_all = max(total[:-1].sum(), 1.)
        total = np.maximum(total, 1.)
        fn = self.matrix[-1, :].sum()
        fp = self.matrix[:, -1].sum()
        acc = self.matrix[np.arange(self.matrix.shape[0]), np.arange(self.matrix.shape[1])]

        missed = self.matrix[-1, :] / total
        misclassed = (total - self.matrix[-1, :] - acc)
        misclassed_all = misclassed[:-1].sum() / total_all
        misclassed /= total
        acc_all = acc[:-1].sum() / total_all
        acc /= total

        if names is None:
            names = [f'{d}' for d in range(self.matrix.shape[0] - 1)]
        else:
            names = list(names)

        len_name = max([len(x) for x in names])
        LOGGER.info(f'{" ":<{len_name}}   {"Acc":>6}  {"MisC":>6}  {"Missed":>6}')
        for idx in range(len(total) - 1):
            LOGGER.info(f'{names[idx]:>{len_name}}:  {acc[idx]:.4f}  {misclassed[idx]:.4f}  {missed[idx]:.4f}')

        LOGGER.info(f'accuracy all: {acc_all:.4f}  missclassified all: {misclassed_all:.4f}')
        LOGGER.info(f'false positive: {int(fp):d}\tfalse negative: {int(fn):d}')

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    @plt_settings()
    def plot(self, normalize=True, save_dir='', names=(), on_plot=None, cmap: str = 'Blues', title: str = None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
            cmap (str): custom colormap for plotting
            title (str): title string for confusion matrix plots
        """
        import seaborn as sn

        use_class = self.matrix.sum(0) > 0
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        # only use classes that have annotations
        array = array[:, use_class]
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        # ticklabels = (list(names) + ['background']) if labels else 'auto'
        if labels:
            x_ticklabels = [name for idx, name in enumerate(list(names) + ['background']) if use_class[idx]]
            y_ticklabels = list(names) + ['background']
        else:
            x_ticklabels = y_ticklabels = "auto"

        annotations = []
        for row_idx in range(self.matrix.shape[0]):
            if normalize:
                annotations.append([f'{array[row_idx, col_idx]:.1%}\n{int(self.matrix[row_idx, class_idx]):d}' for
                                    col_idx, class_idx in enumerate(np.nonzero(use_class)[0])])
            else:
                annotations.append([f'{int(self.matrix[row_idx, class_idx]):d}' for
                                    class_idx in np.nonzero(use_class)[0]])
        annotations = np.array(annotations)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       # annot=nc < 30,
                       annot=annotations if nc < 30 else False,
                       annot_kws={
                           'size': 16},     # 8
                       cmap=cmap,
                       # fmt='.2f' if normalize else '.0f',
                       fmt='',
                       square=True,
                       vmin=0.0,
                       xticklabels=x_ticklabels,
                       yticklabels=y_ticklabels).set_facecolor((1, 1, 1))
        if title is None:
            title = 'Confusion Matrix' + ' Normalized' * normalize
            plot_fname = title
        else:
            plot_fname = title + ' Normalized' * normalize
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(title)
        plot_fname = Path(save_dir) / f'{plot_fname.lower().replace(" ", "_")}.png'
        fig.savefig(plot_fname, dpi=250, bbox_inches='tight')
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)
