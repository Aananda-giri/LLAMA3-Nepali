import argparse
from huggingface_hub import hf_hub_download, HfFileSystem
from kaggle_secrets import UserSecretsClient
import os

from functions import get_max_global_step_file

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")

def download_checkpoint(checkpoint_name, destination):
    print('-'*50, f'\nDownloading checkpoint: \"{checkpoint_name}\" to folder: \"{destination}\"...\n', '-'*50)
        
    # make output directory if not exists
    os.makedirs(destination, exist_ok=True)

    # e.g. download link
    # https://huggingface.co/Aananda-giri/LLAMA3-Nepali/resolve/main/model_checkpoints/model_pg_10000_steps.pth

    # download the checkpoint
    # os.system(f'wget https://huggingface.co/Aananda-giri/LLAMA3-Nepali/resolve/main/{checkpoint_name} -O {destination}/{checkpoint_name}')

    # downloads to ./
    # maintain original file structure: model_checkpoints/model_pg_10000_steps.pth
    hf_hub_download(repo_id="Aananda-giri/LLAMA3-Nepali", filename=checkpoint_name, local_dir=destination, token=hf_token)
    
    print(f'Checkpoint downloaded to: {destination}/{checkpoint_name}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='download_checkpoint_from_hub Configuration')
    # parser.add_argument('--checkpoint_name', type=str, default='parameters_300m/model_pg_193000_steps.pth',
    #                     help='name of model checkpoint. e.g. model_checkpoints/model_pg_10000_steps.pth')
    parser.add_argument('--checkpoint_path', type=str, default='parameters_300m',
                        help='name of model checkpoint. e.g. parameters_300m')
    parser.add_argument('--destination', type=str, default='destination',
                        help='destination path to save the checkpoint. e.g. /kaggle/working/')
    args = parser.parse_args()

    fs = HfFileSystem(token=hf_token)
    # hf file names 
    all_file_names = fs.ls(f"Aananda-giri/LLAMA3-Nepali/{args.checkpoint_path}/", detail=False)
    latest_checkpoint = get_max_global_step_file(files_list=all_file_names) # e.g. 'Aananda-giri/LLAMA3-Nepali/parameters_300m/model_pg_193000_steps.pth'
    latest_checkpoint = latest_checkpoint.split('LLAMA3-Nepali/')[1]    # e.g. 'parameters_300m/model_pg_193000_steps.pth'
    print(f'latest_checkpoint: {latest_checkpoint}')
    download_checkpoint(latest_checkpoint, args.destination)
    
    
