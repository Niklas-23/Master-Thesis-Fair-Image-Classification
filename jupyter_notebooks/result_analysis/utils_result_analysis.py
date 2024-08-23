import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
plt.rcParams.update({'font.size': 12})


def plot_metric(df_tuples, metric_name, y_label=None, title=None, smoothing_factor=0, save_fig_path=None, fontsize=12):
    if not 0 <= smoothing_factor <= 1:
        raise ValueError("The smoothing factor must be between 0 (inclusive) and 1 (inclusive).")

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(df_tuples)))

    for idx, (df, name) in enumerate(df_tuples):
        if metric_name in df.columns:
            if smoothing_factor > 0:
                # If the smoothing factor is greater than 0, show the original line lightly
               #plt.plot(df[metric_name], label=name, color=colors[idx], alpha=0.3)
                # Calculate the trend line with exponential moving average (EMA)
                smoothed_values = df[metric_name].ewm(alpha=1 - smoothing_factor, adjust=False).mean()
                #plt.plot(smoothed_values, color=colors[idx], linestyle='--', label=name + ' (smoothed)')
                plt.plot(smoothed_values, color=colors[idx], label=name)
            else:
                plt.plot(df[metric_name], label=name, color=colors[idx])
        else:
            print(f"Warning: '{metric_name}' not found in DataFrame '{name}'")
    if title is None:
        if smoothing_factor == 0:
            plt.title(f'Trend of Metric "{metric_name}"')
        else:
            plt.title(f'Trend of Metric "{metric_name}" (Smoothing Factor: {smoothing_factor})')
    else:
        plt.title(title, fontsize=fontsize)
    plt.xlabel('Epoch', fontsize=fontsize )
    if y_label is not None:
        plt.ylabel(y_label, fontsize=fontsize)
    else:
        plt.ylabel(metric_name, fontsize=fontsize)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize)
    #plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(f"../dataset_plots/{save_fig_path}.png", dpi=250)
    plt.show()


def get_df(model, dataset_name, version, plot_label=None):
    if plot_label is None:
        return pd.read_csv(
            f"../../model_weights/{model}/{dataset_name}/{version}/metrics.csv"), f"{dataset_name}/{model}/{version}"
    else:
        return pd.read_csv(f"../../model_weights/{model}/{dataset_name}/{version}/metrics.csv"), plot_label
