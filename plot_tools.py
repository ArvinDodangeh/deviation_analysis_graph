import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class PlotTools:
    def __init__(self, plot_dir='plots'):
        self.plot_dir = plot_dir
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def draw_plots(self, json_path):
        try:
            df = pd.read_json(json_path)
        except ValueError as e:
            print(f"Error reading JSON file: {e}")
            return []

        plots_path = []

        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df, x='gt_corners', y='rb_corners', alpha=0.2)
        sns.lineplot(x=[df['gt_corners'].min(), df['gt_corners'].max()],
                     y=[df['gt_corners'].min(), df['gt_corners'].max()],
                     color='blue')
        plt.xlabel('Ground Truth Corners')
        plt.ylabel('Predicted Corners')
        plt.title('Ground Truth vs Predicted Corners')
        plot_path = os.path.join(self.plot_dir, 'gt_vs_rb_corners.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plots_path.append(plot_path)
        plt.close()

        plt.figure(figsize=(10, 6))
        df[['mean', 'max', 'min']].plot(kind='box')
        plt.title('Distribution of Deviation Statistics')
        plt.ylabel('Deviation (degrees)')
        plot_path = os.path.join(self.plot_dir, 'distribution_of_deviation_statistics.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plots_path.append(plot_path)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(df['floor_mean'], df['ceiling_mean'])
        plt.title('Floor vs. Ceiling Mean Deviation')
        plt.xlabel('Floor Mean Deviation (degrees)')
        plt.ylabel('Ceiling Mean Deviation (degrees)')
        plt.plot([0, max(df['floor_mean'].max(), df['ceiling_mean'].max())],
                 [0, max(df['floor_mean'].max(), df['ceiling_mean'].max())],
                 color='red', linestyle='--')
        plot_path = os.path.join(self.plot_dir, 'floor_mean_vs_ceiling_mean.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plots_path.append(plot_path)
        plt.close()

        plt.figure(figsize=(12, 10))
        corr = df[['mean', 'max', 'min', 'floor_mean', 'floor_max', 'floor_min', 'ceiling_mean', 'ceiling_max',
                   'ceiling_min']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Deviation Metrics')
        plot_path = os.path.join(self.plot_dir, 'correlation_heatmap.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plots_path.append(plot_path)
        plt.close()

        sns.pairplot(df[['mean', 'max', 'min', 'floor_mean', 'floor_max', 'floor_min', 'ceiling_mean', 'ceiling_max',
                         'ceiling_min']])
        plt.suptitle('Pair Plot of Deviation Metrics', y=1.02)
        plot_path = os.path.join(self.plot_dir, 'pair_plot.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plots_path.append(plot_path)
        plt.close()

        return plots_path
