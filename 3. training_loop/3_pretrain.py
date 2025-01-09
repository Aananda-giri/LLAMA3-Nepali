# modified code from https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-D/01_main-chapter-code/appendix-D.ipynb

import argparse
import math
import os
from pathlib import Path
import time


# modified. tokenizer import
# import tiktoken
from transformers import PreTrainedTokenizerFast
import torch
from previous_chapters import (
    # create_dataloader_v2, # modified. use create_dataloader_v3 instead
    create_dataloader_v3,
    Llama3Model,
    generate_and_print_sample,
    calc_loss_batch,
    evaluate_model,
    plot_losses,
    Tokenizer,
    ChatFormat
)

from functions import delete_checkpoints_except_n_highest_steps, get_max_global_step_file, format_time_elapsed, push_latest_checkpoint_to_hub
from debug_dataloaders import create_debug_dataloaders


def create_dataloaders(batch_size, num_workers):
    ''' 
    modified. sebastian
    parameter: text_data is removed
    parameter: max_length, stride are removed
    
    modified. GPT2
    parameter: train_ratio: removed (data is pre-split in train and test in hf data (since it takes a while to split))
    '''
    train_loader, val_loader = create_dataloader_v3(
        batch_size=batch_size,
        shuffle=False,  # modified. to avoid  shuffling the data
        drop_last=True,
        num_workers=num_workers
        # context_length=args.context_length
    )
    return train_loader, val_loader


def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)



def get_lr(initial_lr, min_lr, peak_lr, global_step, warmup_steps, lr_increment, const_min_lr_steps):
     # Adjust the learning rate based on the current phase (warmup or cosine annealing)
    # 1) Linear warmup
    if global_step < warmup_steps:
        lr = initial_lr + global_step * lr_increment  
        return lr
    # 2) Cosine annealing after warmup
    elif global_step < const_min_lr_steps:
        
        progress = ((global_step - warmup_steps) / (const_min_lr_steps - warmup_steps)) # modified. to smoothen the curve original: progress = ((global_step - warmup_steps) / (total_training_steps - warmup_steps))
        lr = min_lr + (peak_lr - min_lr) * 0.5 * (
            1 + math.cos(math.pi * progress))
        return lr
    # 3) constant minumum learning rate
    else:
        return min_lr


BOOK_VERSION = True



def train_model(model, train_loader, val_loader, optimizer, device,
            n_epochs, eval_freq, eval_iter, start_context, output_dir, tokenizer,
            previous_global_step=None, train_losses = [], val_losses=[], track_tokens_seen=[],
            track_lrs=[], previous_epochs = 0
    ):
    
    initial_lr=args.initial_lr
    min_lr=args.min_lr

    print("Training ...")
    # train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
    
    # modified. for resuming
    train_loader_index = -1
    train_loader_resume_index = previous_global_step % len_train_loader if previous_global_step else -1

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # # Calculate the total number of iterations in the training process
    # total_training_steps = len_train_loader * n_epochs# len(train_loader) * n_epochs
    
    total_steps = len_train_loader * args.n_epochs
    print(f'total_steps: {total_steps}')
    warmup_steps = int(0.2 * total_steps) # 20% warmup
    print(f' warmup_steps: {warmup_steps}')

    # modified. use constant min_lr for last 10% of training data
    const_min_lr_steps = int(.9 * total_steps)
    print(f' constant min_lr after: {const_min_lr_steps} steps')
    
    
    # (calclulate initially) Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps
    try:
        
        evaluated_once = False  # modified. to evaluate the model at least once (at the start)
        done_resume = False # modified. to check if the resume script has been run once
        pushed_to_hub_once = False
        push_to_hub_seconds = args.push_to_hub_hours * 3600 # seconds so that we dont have to calculate difference in hours by dividing by 3600 at every step.
        
        print(f'push to hub once every {args.push_to_hub_hours} hours i.e. {push_to_hub_seconds} seconds..')
        for epoch in range(n_epochs):
            model.train()   # Training mode
            print(f'skipping train_loader till index: {train_loader_resume_index}... ', end = '')
            for input_batch, target_batch in train_loader:
                
                global_step += 1
                train_loader_index += 1    # previous_global_step % len(train_loader)

                # modified. added to resume feature
                if not done_resume and previous_global_step:
                    if train_loader_index < train_loader_resume_index:
                        # naive implementation.
                        # to iterate through train_loader until train_loader_index gets to train_loader_resume_index

                        # train_loader_index += 1    # previous_global_step % len(train_loader)
                        # print('.', end = '')
                        continue    # continue train_loader till global_step gets to previous_global_step
                    
                    # this code is supposed to runs only once (at the end of skipping dataloaders)
                    done_resume = True
                    global_step = previous_global_step
                    print('done.')
                    print('\n' + '-'*70 + '\n')
                    time_elapsed = format_time_elapsed(start_time, time.time())
                    print(f"\n{'-'*70}\n resuming from global_step : {global_step} \n train_loader_index: {train_loader_index} \n len_train_loader: {len_train_loader} \n Time: {time_elapsed}", end = '\n' + '-'*70 + '\n')
                
                optimizer.zero_grad()

                # modified. added constant minimum learning rate along with linear_warmup+cosine_decay
                lr = get_lr(initial_lr, min_lr, peak_lr, global_step, warmup_steps, lr_increment, const_min_lr_steps)

                # Apply the calculated learning rate to the optimizer
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                track_lrs.append(lr)  # Store the current learning rate

                # Calculate and backpropagate the loss
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()

                # Apply gradient clipping after the warmup phase to avoid exploding gradients

                
                if not args.debug:
                    '''
                    * Gradient clipping might be unnecessary during this warm-up because gradients tend to be smaller.
                    '''
                    if BOOK_VERSION:
                        if global_step > warmup_steps:
                            # Triggered After completing the warm-up phase (dont know why this matters. it was implemented by sebastian)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                    else:
                        if global_step >= warmup_steps:  # the book originally used global_step > warmup_steps, which lead to a skipped clipping step after warmup
                            # Triggered During and after the last warm-up step
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                tokens_seen += input_batch.numel()

                if global_step % eval_freq == 0 or not evaluated_once:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader,
                        device, eval_iter, len_train_loader, len_val_loader
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    # Print the current losses
                    time_elapsed = format_time_elapsed(start_time, time.time())
                    print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}"
                        f" Time: {time_elapsed}"
                    )
                    evaluated_once = True
                
                # Save at every 10,000 steps
                if global_step % args.save_ckpt_freq_steps == 0 and global_step != 0:
                    delete_checkpoints_except_n_highest_steps(n=1)  # modified. to delete the previous steps checkpoint#
                    save_file_path = os.path.join(output_dir, f"model_pg_{global_step}_steps.pth")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "track_tokens_seen": track_tokens_seen,
                        "track_lrs": track_lrs,
                        "epochs": global_step % len_train_loader if global_step > len_train_loader else 0,
                        "global_step": global_step +1,  # +1 because next `global_step` will be incremented by 1 and we will set: next `global_step = previous_global_step``
                        },
                        save_file_path
                    )
                    print(f"Saved {save_file_path}")
                    # Generate and print a sample from the model to monitor progress (at the end of each epoch)
                    generate_and_print_sample(PROMPT="रामले भात", tokenizer=tokenizer, chat_tokenizer=chat_tokenizer, model=model, device=device, context_length = LLAMA32_CONFIG["context_length"])
                    # generate_and_print_sample(
                    #     model, tokenizer, device, start_context
                    # )
                
                # push latest checkpoint to hub (once every 11 hours)
                time_elapsed = (time.time() - start_time)
                if time_elapsed > push_to_hub_seconds and not pushed_to_hub_once:
                    push_latest_checkpoint_to_hub()
                    pushed_to_hub_once = True
                    
            # Save at the end of each epoch
            delete_checkpoints_except_n_highest_steps(n=1)  # modified. to delete the previous steps checkpoint
            new_epochs = global_step % len_train_loader if global_step > len_train_loader else 0
            save_file_path = os.path.join(output_dir, f"model_pg_epoch_{new_epochs}.pth")
            torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "track_tokens_seen": track_tokens_seen,
                    "track_lrs": track_lrs,
                    "epochs": new_epochs,
                    "global_step": global_step +1,  # +1 because next `global_step` will be incremented by 1 and we will set: next `global_step = previous_global_step``
                    },
                    save_file_path
            )
            print(f"Saved {save_file_path}")
    except KeyboardInterrupt:
        file_name = os.path.join(output_dir, f"model_pg_{global_step}_interrupted.pth")
        # modified. to save optimizer state_dict along with model state dict
        # torch.save(model.state_dict(), file_name)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "track_tokens_seen": track_tokens_seen,
            "track_lrs": track_lrs,
            "epochs": global_step % len_train_loader if global_step > len_train_loader else 0,
            "global_step": global_step,
            }, 
            file_name
        )
        print(f"Saved {file_name}")

    return train_losses, val_losses, track_tokens_seen, track_lrs



    

if __name__ == "__main__":
    # Note:
    # Uncomment the following code to calculate the execution time
    global start_time
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='LLAMA3.2 Model Training Configuration')
    

    parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_iter', type=int, default=1000,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=100_000,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--peak_lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer') # this was originally set to 5e-4 in the book by mistake correction: 0.001
    parser.add_argument('--initial_lr', type=float, default=1e-5,
                        help='Learning rate for the optimizer') # this was originally set to 5e-4 in the book by mistake correction: 0.001
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Learning rate for the optimizer') # this was originally set to 5e-4 in the book by mistake correction: 0.001
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--debug', type=str, default="False",
                        help='Uses a very small model for debugging purposes')
    parser.add_argument('--compile_model', type=str, default="True", # modified. added
                        help='whether or not to compile the model')
    parser.add_argument('--max_text_len', type=int, default=45000000,
                        help='testing different text sizes.')
    
    
    
    # modified. added resume_from_previous_training
    parser.add_argument('--resume_from_previous_training', type=str, default="True",
                        help='whether or not to resume from saved previous training checkpoint')
    parser.add_argument('--push_to_hub_hours', type=float, default=11.5,
                        help='how often to push to hub in number of steps.')
    parser.add_argument('--save_ckpt_freq_steps', type=int, default=10_000,
                        help='how often to save the model checkpoint in steps')
    parser.add_argument('--context_length', type=int, default=1024,
                        help='context length (default: 1024)')

    args = parser.parse_args()
    args.resume_from_previous_training = args.resume_from_previous_training.lower() == 'true'
    args.compile_model = args.compile_model.lower() == 'true'
    args.debug = args.debug.lower() == 'true'
    torch.manual_seed(123)
    
    
    # modified. code to load the tokenizer
    # tokenizer = tiktoken.get_encoding("gpt2")
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("Aananda-giri/NepaliBPE")
    tokenizer = Tokenizer("tokenizer.json")
    chat_tokenizer = ChatFormat(tokenizer)
    
    if args.debug:
        print(f'---------------------\nDEBUG MODE\n---------------------')
        # Debug mode
        LLAMA32_CONFIG = {
            # d_out = emb_dim
            # Embedding dimension <d_out // num_heads> must be even
            "vocab_size": 50006,      # <len(tokenizer.tokenizer)=50006> Vocabulary size
            "context_length": 10,  # Context length
            # d_in=d_out=emb_dim,
            # d_out must be divisible by num_heads
            "emb_dim": 8,            # Embedding dimension
            # (num_heads must be divisible by num_kv_groups)
            "n_heads": 4,              # Number of attention heads
            "n_layers": 2,             # Number of layers
            "hidden_dim": 16,         # Size of the intermediate dimension in FeedForward
            "n_kv_groups": 2,           # Key-Value groups for grouped-query attention
            "rope_base": 500_000.0,     # The base in RoPE's "theta"
            "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
            "rope_freq": {              # RoPE frequency scaling
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_context_length": 8192,
            }
        }

        # Custom dataloader for debug mode
        # --------------------------------
        def read_text_file(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()
            return text_data
        text_data = read_text_file("cleaned_bhagavad_gita_data.txt") + " <|endoftext|> "
        train_loader, val_loader = create_debug_dataloaders(
            text_data,
            tokenizer,
            train_ratio=0.9,
            batch_size=2,
            max_length=LLAMA32_CONFIG["context_length"],
            stride=LLAMA32_CONFIG["context_length"],
            num_workers=0
        )
        
    else:
        # 
        print(f'---------------------\nDEBUG MODE=False\n---------------------')
        # Llama 3.2 200M
        LLAMA32_CONFIG = {
            "vocab_size": 50006,       # <len(tokenizer.tokenizer)=50006> 128_256 reduced vocabulary size
            "context_length": 512,      # 131_072 reduced Context length (unrelated to model size but higheer context length consumes more RAM)
            "emb_dim": 1024,            # 2048 reduced Embedding dimension
            "n_heads": 16,              # 32 reduced Number of attention heads
            "n_layers": 8,             # 16 reduced Number of layers
            "hidden_dim": 4096,         # 8192 Size of the intermediate dimension in FeedForward
            "n_kv_groups": 8,           # 8 Key-Value groups for grouped-query attention
            "rope_base": 500_000.0,     # 500_000 The base in RoPE's "theta"
            "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
            "rope_freq": {              # RoPE frequency scaling
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_context_length": 8192,
            }
        }

        # Initialize new data loader
        train_loader, val_loader = create_dataloaders(
            # train_ratio=0.9,
            batch_size=args.batch_size,
            num_workers=0 
        )
    
    LLAMA_SIZE_STR = "2M" if args.debug else "200M"
    
    def set_loader_lengths(debug, train_loader=None, val_loader=None):
        if debug:
            len_train_loader = len(train_loader)
            len_val_loader = len(val_loader)
        else:
            len_train_loader = int(4781060 / args.batch_size)    # train data contains 4781060 rows
            len_val_loader = int(531229 / args.batch_size)
        return len_train_loader, len_val_loader

    global len_train_loader
    global len_val_loader

    len_train_loader, len_val_loader = set_loader_lengths(
        args.debug, train_loader, val_loader
    )
    # re-scaling theta
    # ------------------------------------------------------------
    
    old_context_length = 131_072    # original context length of llama3.2 model
    new_context_length = LLAMA32_CONFIG["context_length"]  # 512 our new context length

    def rescale_theta(theta_old, context_length_old, context_length_new):
        # # linear scaling by sebastian
        # scaling_factor = context_length_new / context_length_old
        
        '''
            Using square root scaling (instead of linear scaling as done by sebastian),
            because linear scaling is resulting in very small theta value.
            which is slowing the training (slower decrease in loss)
            might be because of the large difference in context length (137_072 vs 512)
        '''
        scaling_factor = math.sqrt(context_length_new/context_length_old)
        theta_new = theta_old * scaling_factor
        return theta_new

    LLAMA32_CONFIG["rope_base"] = rescale_theta(
        LLAMA32_CONFIG["rope_base"],
        old_context_length,
        new_context_length
    )

    print("New RoPE theta (i.e. LLAMA32_CONFIG[\"rope_base\"]):", LLAMA32_CONFIG["rope_base"])
    

    model = Llama3Model(LLAMA32_CONFIG)
    # compile the model
    if args.compile_model:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    
    # Check buffers
    # --------------
    print('The following is expected to print True to confirm buffers are reused instead of being (wastefully) recreated:')
    print(model.trf_blocks[0].att.mask is model.trf_blocks[-1].att.mask)
    print(model.trf_blocks[0].att.cos is model.trf_blocks[-1].att.cos)
    print(model.trf_blocks[0].att.sin is model.trf_blocks[-1].att.sin)

    # Display number of parameters
    # -----------------------------
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    # Account for weight tying
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")
    
    # Display model_memory_size
    # -----------------------------------------------------------------------
    def model_memory_size(model, input_dtype=torch.float32):
        total_params = 0
        total_grads = 0
        for param in model.parameters():
            # Calculate total number of elements per parameter
            param_size = param.numel()
            total_params += param_size
            # Check if gradients are stored for this parameter
            if param.requires_grad:
                total_grads += param_size

        # Calculate buffer size (non-parameters that require memory)
        total_buffers = sum(buf.numel() for buf in model.buffers())

        # Size in bytes = (Number of elements) * (Size of each element in bytes)
        # We assume parameters and gradients are stored in the same type as input dtype
        element_size = torch.tensor(0, dtype=input_dtype).element_size()
        total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

        # Convert bytes to gigabytes
        total_memory_gb = total_memory_bytes / (1024**3)

        return total_memory_gb

    print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")
    # -----------------------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    print(f'device: {device}')

    
    
    # model = GPTModel(GPT_CONFIG_124M)
    # model.to(device)
    peak_lr = args.peak_lr # 0.001  # this was originally set to 5e-4 in the book by mistake
    weight_decay = 0.1 if args.debug else 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)  # the book accidentally omitted the lr assignment
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # global_step=0
    
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    # previous_epochs = 0
    previous_global_step = None
    # this should work for epochs but epochs take a long time to train (so were sabing for every 10,000 steps)
    # latest_model_checkpoint = get_max_epoch_file(directory='model_checkpoints')
    latest_model_checkpoint = get_max_global_step_file(directory='model_checkpoints')
    
    # if args.load_model and os.path.exists(output_dir):
    print(f'\n\nargs.resume_from_previous_training: {args.resume_from_previous_training}\n\n')
    if latest_model_checkpoint and args.resume_from_previous_training:
        
        print(f'Loading existing model: {latest_model_checkpoint}', end = '\n' + '-'*70 + '\n')
        
        checkpoint = torch.load(latest_model_checkpoint, weights_only=False)
        
        # modified (added model loading code)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)  # the book accidentally omitted the lr assignment
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        train_losses = checkpoint["train_losses"]
        print(f'train_losses: {type(train_losses)}  len: {len(train_losses)}')

        val_losses = checkpoint["val_losses"]
        print(f'val_losses: {type(val_losses)}  len: {len(val_losses)}')

        track_tokens_seen = checkpoint["track_tokens_seen"]
        print(f'track_tokens_seen: {type(track_tokens_seen)}  len: {len(track_tokens_seen)}')

        track_lrs = checkpoint["track_lrs"]
        print(f'track_lrs: {type(track_lrs)}  len: {len(track_lrs)}')

        previous_epochs = checkpoint["epochs"]
        print(f'previous epochs: {type(previous_epochs)} {previous_epochs}')

        previous_global_step = checkpoint["global_step"]
        print(f'previous global step: {previous_global_step} \n previous epochs: {previous_epochs}')
        print(end = '\n' + '-'*70 + '\n')
        
    else:
        print(f'starting new model from scratch')

    # modified
    # n_epochs = 15
    n_epochs = args.n_epochs

    # data_dir = args.data_dir
    # all_files = [os.path.join(path, name) for path, subdirs, files
    #              in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    # total_files = len(all_files)

    # if total_files == 0:
    #     print("No training text files found. Make sure you "
    #           "selected the correct input directory")
    #     quit()
    # print("Total files:", total_files)

    # for index, file_path in enumerate(all_files, 1):
    # book_start_time = time.time()
    # text_data = read_text_file(file_path) + " <|endoftext|> "
    # text_data = text_data[:args.max_text_len]
    # print(f"Tokenizing file {index} of {total_files}: {file_path}")
    
    print(f'len. train_loader: {len_train_loader}')
    print(f'len.val_loader: {len_val_loader}')  # len(val_loader)
    
    train_losses, val_losses, track_tokens_seen, track_lrs = train_model(
        model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
        eval_freq=args.eval_freq, eval_iter=1, start_context="रामले भात", # "Every effort moves you", <modified>
        output_dir=output_dir, tokenizer=tokenizer, previous_global_step=previous_global_step,
        train_losses = train_losses, val_losses=val_losses, track_tokens_seen=track_tokens_seen, track_lrs=track_lrs,
        # previous_epochs = previous_epochs
        
    )
    epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
    plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses, output_dir)

    # print_eta(start_time, book_start_time, index, total_files)

    
    # modified. to save optimizer state_dict along with model state dict
    # torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    
    # lets save at the end of each epoch instead
    # torch.save({
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    #     "train_losses": train_losses,
    #     "train_losses": train_losses,
    #     "track_tokens_seen": track_tokens_seen,
    #     "track_lrs": track_lrs,
    #     "epochs": n_epochs + previous_epochs,
    #     }, 
    #     output_dir / "model_pg_final.pth"
    # )
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Note:
    # Uncomment the following code to show the execution time
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")# code from https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-D/01_main-chapter-code/appendix-D.ipynb
