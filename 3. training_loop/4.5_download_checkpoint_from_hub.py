import argparse
from huggingface_hub import hf_hub_download
from kaggle_secrets import UserSecretsClient
import os

def download_checkpoint(checkpoint_name, destination):
    print('-'*50, f'\nDownloading checkpoint: \"{args.checkpoint_name}\" to folder: \"{destination}\"...\n', '-'*50)
        
    # make output directory if not exists
    os.makedirs(destination, exist_ok=True)

    # e.g. download link
    # https://huggingface.co/Aananda-giri/LLAMA3-Nepali/resolve/main/model_checkpoints/model_pg_10000_steps.pth

    # download the checkpoint
    # os.system(f'wget https://huggingface.co/Aananda-giri/LLAMA3-Nepali/resolve/main/{checkpoint_name} -O {destination}/{checkpoint_name}')
    
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")

    # downloads to ./
    # maintain original file structure: model_checkpoints/model_pg_10000_steps.pth
    hf_hub_download(repo_id="Aananda-giri/LLAMA3-Nepali", filename=checkpoint_name, local_dir=destination, token=hf_token)
    
    print(f'Checkpoint downloaded to: {destination}/{checkpoint_name}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='download_checkpoint_from_hub Configuration')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_name',
                        help='name of model checkpoint. e.g. model_checkpoints/model_pg_10000_steps.pth')
    parser.add_argument('--destination', type=str, default='destination',
                        help='destination path to save the checkpoint. e.g. /kaggle/working/')
    args = parser.parse_args()

    download_checkpoint(args.checkpoint_name, args.destination)
    
    
