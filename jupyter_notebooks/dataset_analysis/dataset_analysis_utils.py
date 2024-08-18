
import pandas as pd
import matplotlib.pyplot as plt

def plot_attribute_protected_distribution(dataframe, attribute_column, gender_column):
    label_distribution = dataframe.groupby(gender_column)[attribute_column].value_counts(normalize=True) * 100
    label_distribution = label_distribution.unstack()

    #plt.figure(figsize=(20, 46))

    ax = label_distribution.plot(kind='bar', color=['darkblue', 'lightblue'])
    plt.title(f'{attribute_column} Distribution by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.xticks([0, 1], ['Female', 'Male'])
    plt.legend([f'Not {attribute_column}', attribute_column], loc='upper right')

    for i in range(len(label_distribution.index)):
        for j in range(len(label_distribution.columns)):
            plt.text(i + j * 0.2 - 0.1, label_distribution.iloc[i, j] + 0.5, f'{label_distribution.iloc[i, j]:.1f}%', ha='center')

    plt.tight_layout()
    plt.show()


def plot_protected_feature_distribution_combined(dataframe, feature_column, protected_column):
    distribution = dataframe.groupby(protected_column)[feature_column].value_counts(normalize=True).unstack()
    counts = dataframe.groupby([protected_column, feature_column]).size().unstack(fill_value=0)

    # Plotting
    ax = distribution.plot(kind='bar', stacked=True)
    plt.xlabel(protected_column)
    plt.ylabel('Proportion')
    x_labels = distribution.index.tolist()
    x_ticks = range(len(x_labels))
    plt.xticks(x_ticks,
               x_labels if len(distribution) <= 10 else ["" if i % 4 != 0 else x_labels[i] for i in x_ticks],
               rotation=45)
    plt.title(f'Distribution of Target Feature {feature_column} by Protected Feature {protected_column}')
    handles, labels = ax.get_legend_handles_labels()

    # Adjust legend labels
    plt.legend(handles, labels, title=feature_column, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Annotate the bars with the absolute number of samples
    for i in range(len(x_labels)):
        for j, bar in enumerate(ax.containers):
            height = bar[i].get_height()
            count = counts.iloc[i, j]
            ax.text(bar[i].get_x() + bar[i].get_width() / 2, height, f'{count}', ha='center', va='bottom')

    plt.show()


def plot_protected_feature_distribution(dataframe, feature_column, protected_column, title=None, save_fig_path=None):
    label_distribution = dataframe.groupby(protected_column)[feature_column].value_counts(normalize=True) * 100
    label_distribution = label_distribution.unstack()

    #plt.figure(figsize=(20, 46))

    ax = label_distribution.plot(kind='bar')
    plt.title(f'{feature_column} Distribution by {protected_column}')
    plt.xlabel('Gender')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.xticks([0, 1], ['Female', 'Male'])
    plt.legend(loc='upper right')
    if title is not None:
        plt.title(title)
    #for i in range(len(label_distribution.index)):
    #    for j in range(len(label_distribution.columns)):
     #       ax.text(i + j * 0.2 - 0.1, label_distribution.iloc[i, j] + 0.5, f'{label_distribution.iloc[i, j]:.1f}%', ha='center')

    #plt.legend(['Not Smiling', 'Smiling'], loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(f"../dataset_plots/{save_fig_path}.png", dpi=250)
    plt.show()


def plot_protected_feature_distribution_binary(dataframe, feature_column, protected_column, title=None, save_fig_path=None):
    label_distribution = dataframe.groupby(protected_column)[feature_column].value_counts(normalize=True) * 100
    label_distribution = label_distribution.unstack()

    #plt.figure(figsize=(20, 46))

    ax = label_distribution.plot(kind='bar')
    plt.title(f'{feature_column} Distribution by {protected_column}')
    plt.xlabel('Gender')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.xticks([0, 1], ['Female', 'Male'])
    plt.legend(loc='upper right')
    if title is not None:
        plt.title(title)

    for i in range(len(label_distribution.index)):
        for j in range(len(label_distribution.columns)):
            if j == 1:
                plt.text((i + j * 0.2) - 0.08, label_distribution.iloc[i, j] + 0.5, f'{label_distribution.iloc[i, j]:.1f}%', ha='center')
            else:
                plt.text((i + j * 0.2) - 0.12, label_distribution.iloc[i, j] + 0.5,
                         f'{label_distribution.iloc[i, j]:.1f}%', ha='center')

    plt.legend([f'Not {feature_column}', feature_column], loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if save_fig_path is not None:
        plt.savefig(f"../dataset_plots/{save_fig_path}.png", dpi=250)

    plt.show()


def plot_target_distribution(dataframe, feature_column, title=None, save_fig_path=None):
    total_distribution = dataframe[feature_column].value_counts(normalize=True)
    total_distribution = total_distribution.sort_index()

    ax = total_distribution.plot(kind='bar')
    plt.xlabel(feature_column)

    x_labels = total_distribution.index.tolist()
    x_ticks = range(len(x_labels))
    plt.xticks(x_ticks,
               x_labels if len(total_distribution) <= 10 else ["" if i % 4 != 0 else x_labels[i] for i in
                                                                              x_ticks], rotation=45)

    for i, v in enumerate(total_distribution):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    max_value = total_distribution.max()
    plt.ylim(0, max_value + 0.05)
    plt.ylabel('Percentage')
    plt.title(f'Total Distribution of Feature {feature_column}')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(f"../dataset_plots/{save_fig_path}.png", dpi=250)

    plt.show()