import os
import re

from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

def delete_checkpoints_except_n_highest_steps(folder_path="model_checkpoints", n=1):
    """
    Deletes all files in the given folder except for the n files with the highest steps.

    Args:
        folder_path (str): Path to the folder containing the files.
        n (int): Number of files to keep.
    """
    # Get a list of all files in the folder
    files = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename)) and filename.startswith("model_pg_") and filename.endswith("_steps.pth")]
    interrupted_files = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename)) and filename.startswith("model_pg_") and filename.endswith("_interrupted.pth")]
    
    # print (f'all files: {files}')
    

    # Extract the step number from each file name and store it in a dictionary
    file_steps = {}
    for file in files:
        match = re.search(r'model_pg_(\d+)_steps\.pth', file)
        if match:
            steps = int(match.group(1))
            file_steps[file] = steps

    # Sort the files by their step numbers in descending order
    sorted_files = sorted(file_steps, key=file_steps.get, reverse=True) # short in descending order
    files_to_delete = sorted_files[n:]
    # print(f'files to delete: {files_to_delete}')
    # return
    
    # Delete all files except for the n files with the highest steps
    for file in files_to_delete:
        os.remove(os.path.join(folder_path, file))
        print(f"Deleted file: {file}")
    
    # Delete all interrupted files
    for file in interrupted_files:
        os.remove(os.path.join(folder_path, file))
        print(f"Deleted file: {file}")



# def get_max_epoch_file(directory='model_checkpoints'):
#     max_epoch = 0
#     max_epoch_file = None
    
#     if os.path.exists(directory):
#         for filename in os.listdir(directory):
#             if filename.startswith("model_pg_epoch_") and filename.endswith(".pth"):
#                 try:
#                     epoch = int(filename.split("_")[-1].split(".")[0])
#                     if epoch > max_epoch:
#                         max_epoch = epoch
#                         max_epoch_file = filename
#                 except Exception as Ex:
#                     print(f'file: {filename} : {Ex}')
    
#     return os.path.join(directory, max_epoch_file) if max_epoch_file else None

def get_max_global_step_file(directory='model_checkpoints'):
    max_step = 0
    max_steps_file = None

    if os.path.exists(directory):
        for filename in os.listdir(directory):
            # format: model_pg_{global_step}_steps.pth
            if filename.startswith("model_pg_") and filename.endswith("_steps.pth"):
                try:
                    step = int(filename.split("model_pg_")[-1].split("_steps.pth")[0])
                    if step > max_step:
                        max_step = step
                        max_steps_file = filename
                except Exception as Ex:
                    print(f'file: {filename} : {Ex}')
    
    return os.path.join(directory, max_steps_file) if max_steps_file else None

from datetime import timedelta
import time

def format_time_elapsed(start_time, end_time):
    # Calculate the time difference
    time_diff = end_time - start_time
    
    # Convert time difference to days, hours, minutes, and seconds
    days = time_diff // (24 * 3600)
    hours = (time_diff % (24 * 3600)) // 3600
    minutes = (time_diff % 3600) // 60
    seconds = time_diff % 60
    
    # Format the elapsed time as a string
    elapsed_time = []
    if days > 0:
        elapsed_time.append(f"{int(days)} day{'s' if days > 1 else ''}")
    if hours > 0:
        elapsed_time.append(f"{int(hours)} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        elapsed_time.append(f"{int(minutes)} minute{'s' if minutes > 1 else ''}")
    if seconds > 0 or not elapsed_time:  # Ensure we always display seconds
        elapsed_time.append(f"{int(seconds)} second{'s' if seconds > 1 else ''}")
    
    # Join all parts and return the result
    return ', '.join(elapsed_time)


def push_latest_checkpoint_to_hub():
    print("-"*50, "\n\tpushing latest model checkpoint to hub\n", "-"*50)

    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")

    # path of latest model checkpoint
    latest_model_checkpoint = get_max_global_step_file(directory='model_checkpoints')   # e.g. 'model_checkpoints/model_pg_20_steps.pth'

    if latest_model_checkpoint:
        print(f'got checkpoint: {latest_model_checkpoint}')
        api = HfApi()

        api.upload_file(
            path_or_fileobj=latest_model_checkpoint,    # e.g. 'model_checkpoints/model_pg_20_steps.pth'
            path_in_repo=latest_model_checkpoint,       # e.g. 'model_checkpoints/model_pg_20_steps.pth'
            repo_id="Aananda-giri/LLAMA3-Nepali",
            repo_type="model",
            token=hf_token
        )
        print(f"model checkpoint pushed to hub... {latest_model_checkpoint}")
    else:
        print("No checkpoint found to push to hub. exiting...")



if __name__ == "__main__":
    # Example usage:
    start_time = time.time()  # Get current time as a timestamp
    time.sleep(5)  # Wait for 5 seconds to simulate elapsed time
    end_time = time.time()  # Get the new timestamp

    print(format_time_elapsed(start_time, end_time))