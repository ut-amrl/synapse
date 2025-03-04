import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.patheffects as pe
from mlxtend.plotting import plot_confusion_matrix
from src.llmgrop.constants import ALL_TASKS


# .user.task.method = [ratings_generations, ...]
PER_BASELINE = 3
PER_OWN = 3
PER_USER = 2
user_demo_data_dir = "test/llmgrop/USER_DEMO_DATA"
user_ratings_dir = "test/llmgrop/USER_RATINGS"


def load_all_data():
    ALL_DATA = edict()
    all_json_files = os.listdir(user_demo_data_dir)
    all_json_files = [f for f in all_json_files if f.endswith(".json")]
    all_names = [f.split(".")[0] for f in all_json_files]

    for user_name in tqdm(all_names, desc="Processing users"):
        user_ratingspath = f"{user_ratings_dir}/{user_name}.json"
        with open(user_ratingspath, "r") as f:
            user_rawdata = edict(json.load(f))

        ALL_DATA[user_name] = edict()
        for task in tqdm(ALL_TASKS.keys(), desc="Processing tasks", leave=False):
            task = task.replace(" ", "").lower()
            ALL_DATA[user_name][task] = edict()
            task_rawdata = user_rawdata[task]

            # baselines ratings
            for p, r in task_rawdata.baselines.items():
                assert r is not None
                method_name = p.split("/")[-2]
                if method_name in ALL_DATA[user_name][task].keys():
                    ALL_DATA[user_name][task][method_name].append(r)
                else:
                    ALL_DATA[user_name][task][method_name] = [r]

            # Synapse ratings
            ALL_DATA[user_name][task].synapse = edict()
            ALL_DATA[user_name][task].synapse[user_name] = list(task_rawdata.own.values())
            for p, r in task_rawdata.users.items():
                ou_name = p.split("/")[-2]
                if ou_name in ALL_DATA[user_name][task].synapse.keys():
                    ALL_DATA[user_name][task].synapse[ou_name].append(r)
                else:
                    ALL_DATA[user_name][task].synapse[ou_name] = [r]
    return ALL_DATA


def eval_baselines(all_data):
    PROCESSED_DATA = edict()
    for m in ['Synapse', 'LLM-GROP', 'LATP', 'GROP', 'TPRA']:
        PROCESSED_DATA[m] = edict({
            "means": [],
            "lows": [],
            "highs": [],
        })
        for tid in range(1, 9):
            task_name = f"task{tid}"
            ratings = []
            for user_name in all_data.keys():
                if m == 'Synapse':
                    ratings += all_data[user_name][task_name].synapse[user_name]
                else:
                    ratings += all_data[user_name][task_name][m]
            PROCESSED_DATA[m].means.append(np.mean(ratings))
            PROCESSED_DATA[m].lows.append(np.mean(ratings) - mean_of_lowest_x_percent(ratings, 55))
            PROCESSED_DATA[m].highs.append(mean_of_highest_x_percent(ratings, 55) - np.mean(ratings))
    return PROCESSED_DATA


def gen_baselines_table(processed_data):
    means_processed_data = {k: v.means for k, v in processed_data.items()}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(means_processed_data, index=np.arange(1, 9))
    df = df.transpose()
    df = df.round(2)

    # Create a table plot using matplotlib
    fig, ax = plt.subplots()

    # Hide the axes
    ax.axis('off')
    ax.axis('tight')

    # Create the table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center',
                     colColours=['#f2f2f2'] * len(df.columns),
                     rowColours=['white'] * len(df))

    # Highlight the maximum values in each column
    for col_idx in range(len(df.columns)):
        max_row_label = df[col_idx + 1].idxmax()  # Get the row label of the max value in this column
        max_row_idx = df.index.get_loc(max_row_label)  # Convert the row label to its numeric index
        cell = table[(max_row_idx + 1, col_idx)]  # Use the numeric index to access the cell
        cell.set_text_props(weight='bold')  # Make the text bold

    plt.show()


def mean_of_highest_x_percent(data, x):
    threshold = np.percentile(data, 100 - x)
    top_values = [value for value in data if value >= threshold]
    mean_top_values = np.mean(top_values)
    return mean_top_values


def mean_of_lowest_x_percent(data, x):
    threshold = np.percentile(data, x)
    bottom_values = [value for value in data if value <= threshold]
    mean_bottom_values = np.mean(bottom_values)
    return mean_bottom_values


def plot_baselines(processed_data):
    tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6', 'Task 7', 'Task 8']
    methods = ['Synapse (ours)', 'LLM-GROP', 'LATP', 'GROP', 'TPRA']

    bar_width = 0.18
    index = np.arange(len(tasks)) * 1.2
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

    plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(10, 4))

    bars0 = ax.bar(index, processed_data['Synapse'].means, bar_width, label=methods[0], yerr=np.array([processed_data['Synapse'].lows, processed_data['Synapse'].highs]), capsize=5, color=colors[0], edgecolor='none', error_kw={'ecolor': 'grey', 'elinewidth': 1})
    bars1 = ax.bar(index + bar_width, processed_data[methods[1]].means, bar_width, label=methods[1], yerr=np.array([processed_data[methods[1]].lows, processed_data[methods[1]].highs]), capsize=5, color=colors[1], edgecolor='none', error_kw={'ecolor': 'grey', 'elinewidth': 1})
    bars2 = ax.bar(index + 2 * bar_width, processed_data[methods[2]].means, bar_width, label=methods[2], yerr=np.array([processed_data[methods[2]].lows, processed_data[methods[2]].highs]), capsize=5, color=colors[2], edgecolor='none', error_kw={'ecolor': 'grey', 'elinewidth': 1})
    bars3 = ax.bar(index + 3 * bar_width, processed_data[methods[3]].means, bar_width, label=methods[3], yerr=np.array([processed_data[methods[3]].lows, processed_data[methods[3]].highs]), capsize=5, color=colors[3], edgecolor='none', error_kw={'ecolor': 'grey', 'elinewidth': 1})
    bars4 = ax.bar(index + 4 * bar_width, processed_data[methods[4]].means, bar_width, label=methods[4], yerr=np.array([processed_data[methods[4]].lows, processed_data[methods[4]].highs]), capsize=5, color=colors[4], edgecolor='none', error_kw={'ecolor': 'grey', 'elinewidth': 1})

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, frameon=False)
    ax.set_xlabel('Tasks', fontsize=12)
    ax.set_ylabel('User Ratings', fontsize=12)
    # ax.set_title('Performance Comparison Across Tasks', fontsize=14)
    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels(tasks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars0, bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height + 0.30),  # Adjusted to be slightly above error bounds
                        xytext=(0, 5),  # Offset for the text
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=10,
                        color=bar.get_facecolor(),
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    for bars in [bars0, bars1, bars2, bars3, bars4]:
        for bar in bars:
            bar.set_linewidth(0)
            bar.set_capstyle('round')  # Slightly rounded edges for bars

    ax.grid(axis='y', color='lightgrey', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"{user_ratings_dir}/baselines_plot.png", dpi=600)


def compute_distance_matrix(matrix):
    """
    Convert a general square matrix to a distance matrix suitable for clustering.
    """
    # Normalize the matrix entries to [0, 1]
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val == 0:
        normalized_matrix = np.zeros_like(matrix)
    else:
        normalized_matrix = (matrix - min_val) / (max_val - min_val)

    # Convert similarities to distances
    distance_matrix = 1 - normalized_matrix
    return distance_matrix


def perform_clustering(distance_matrix):
    """
    Perform hierarchical clustering and return the ordering of items.
    """
    # Ensure the distance matrix is symmetric
    if not np.allclose(distance_matrix, distance_matrix.T):
        # Symmetrize the distance matrix
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Convert the distance matrix to a condensed form
    condensed_distance = squareform(distance_matrix, checks=False)
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method='average')
    # Get the order of items
    order = leaves_list(linkage_matrix)
    return order


def reorder_matrix(matrix, order):
    """
    Reorder a matrix based on the given item order.
    """
    return matrix[np.ix_(order, order)]


def diagonality_score(matrix):
    """
    Compute the diagonality score of a matrix.
    """
    n = matrix.shape[0]
    score = 0.0
    for i in range(n):
        for j in range(n):
            weight = 1.0 / (1.0 + abs(i - j))  # Higher weight for elements near the diagonal
            score += weight * matrix[i, j]
    return score


def diagonalize_mat(mat):
    orig_score = diagonality_score(mat)
    distance_matrix = compute_distance_matrix(mat)
    order = perform_clustering(distance_matrix)
    A = reorder_matrix(mat, order)
    score = diagonality_score(A)
    index_order = [i - 1 for i in order]
    return A, index_order, (orig_score, score)


def plot_userstudy_confusion_mat(all_data):
    all_users = list(all_data.keys())
    N = len(all_users)
    user_processed_data = edict()
    user_processed_data.alltasks = np.zeros((N, N))
    for tid in range(1, 9):
        task_name = f"task{tid}"
        user_processed_data[task_name] = np.zeros((N, N))
        for ir, rater in enumerate(all_users):
            for ip, generator in enumerate(all_users):
                aval_ratings = all_data[rater][task_name].synapse[generator]
                aval_ratings = sorted(aval_ratings, reverse=True)
                aval_ratings = aval_ratings[:PER_USER]
                user_processed_data[task_name][ir, ip] = np.mean(aval_ratings)
        user_processed_data.alltasks += user_processed_data[task_name]
    user_processed_data.alltasks /= 8
    A, index_order, scores = diagonalize_mat(user_processed_data.alltasks)
    print(f"Original score: {scores[0]:.2f}, Diagonalized score: {scores[1]:.2f}")

    # alltasks confmat
    fig2, ax2 = plot_confusion_matrix(conf_mat=A, class_names=['u{}'.format(i) for i in range(10)])
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"{user_ratings_dir}/userstudy_all.png", dpi=600)

    # individual task confmats
    A8 = []
    # order = [i + 1 for i in index_order]
    for tid in range(1, 9):
        task_name = f"task{tid}"
        cur_mat = user_processed_data[task_name]
        # A_task = reorder_matrix(cur_mat, order)
        A_task, *_ = diagonalize_mat(cur_mat)
        A8.append(A_task)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(A8[i], cmap='Blues', interpolation='nearest')
        ax.set_title(f'Task {i+1}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"{user_ratings_dir}/userstudy_tasks.png", dpi=600)


if __name__ == "__main__":
    ALL_DATA = load_all_data()
    PROCESSED_DATA = eval_baselines(ALL_DATA)
    gen_baselines_table(PROCESSED_DATA)
    plot_baselines(PROCESSED_DATA)
    plot_userstudy_confusion_mat(ALL_DATA)
