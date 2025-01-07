from functions import get_max_global_step_file
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient


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

# if __name__ == "__main__":
#     push_latest_checkpoint_to_hub()

if __name__ == "__main__":
    push_latest_checkpoint_to_hub()