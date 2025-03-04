import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

user_demo_data_dir = "test/llmgrop/USER_DEMO_DATA"

# Get all JSON file names and extract user names
all_json_files = os.listdir(user_demo_data_dir)
all_json_files = [f for f in all_json_files if f.endswith(".json")]
all_names = [f.split(".")[0] for f in all_json_files]

# Create a figure for subplots with size adjusted for 5x10 grid
fig, axes = plt.subplots(5, 10, figsize=(20, 10))

for user_idx, user_name in enumerate(tqdm(all_names)):
    dir_path = f"{user_demo_data_dir}/representatives/{user_name}"

    for task_idx in range(5):
        img_path = os.path.join(dir_path, f"Task {task_idx + 1}.png")
        img = Image.open(img_path)

        # Place the image in the correct subplot location
        ax = axes[task_idx, user_idx]
        ax.imshow(img)
        # ax.axis('off')  # Hide axes for a cleaner look
        # Hide only ticks and spines but keep the ylabel visible
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add task label to the first column of each row
        if user_idx == 0:
            ax.set_ylabel(f'Task {task_idx + 1}', rotation=90, labelpad=10, fontsize=15, verticalalignment='center')

# Adjust the layout to make space for the y-axis labels
# plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)
plt.tight_layout()
# plt.show()
plt.savefig(f"{user_demo_data_dir}/representatives/group_users_repr.png", dpi=600)
