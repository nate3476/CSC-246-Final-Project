import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from climb_data import KilterDataset, climb_collate_fn
from encoder import ClimbEncoder
from decoder import ClimbDecoder
from climb_util import show_climb
from train import train_epoch, validate_epoch
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--grade", type=float, required=True)
    parser.add_argument("--prompt", type=int, nargs="*", default=[])
    parser.add_argument("--train", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="models/decoder.pt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--load", action='store_true', default=False)
    parser.add_argument("--metric", action='store_true', default=False)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_dataset = KilterDataset(root='data', download=True)
    training_data, test_data = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

    batch_size = 64
    training_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=climb_collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=climb_collate_fn)

    vocab_size = len(full_dataset.token_dict)
    n_grades = 39

    encoder = ClimbEncoder(vocab_size).to(device)
    decoder = ClimbDecoder(vocab_size=vocab_size, n_grades=n_grades).to(device)
    past_epochs = 0
    if os.path.exists(args.model_path) and (args.train == 0 or args.load == True):
        print(f"Loading model from {args.model_path}")
        loaded_model = torch.load(args.model_path, map_location=device)
        decoder.load_state_dict(loaded_model['model_state_dict'])
        past_epochs = loaded_model['epoch']
        print(f"Loaded model from epoch {loaded_model.get('epoch', 'unknown')}")
    elif args.train == 0:
        print(f"Warning: No saved model found at {args.model_path}. Using untrained model.")

    metric_loss = [] # just for data collection for report
    metric_accuracy = [] # same
    if (args.train > 0):
        print(f"Beginning training for {args.train} epochs...")
        lr = 1e-4
        pad_idx = decoder.pad_idx
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        for i in range(args.train):
            train_loss, train_acc = train_epoch(decoder, training_dataloader, optimizer, criterion, device)
            print(f"Average loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}%, epoch [{i + 1}/{args.train}]")
            metric_loss.append(train_loss)
            metric_accuracy.append(train_acc)
        if(args.metric):
            print("LOSS")
            print(metric_loss)
            print("ACCURACY")
            print(metric_accuracy)
        
        # save the model (I looked up how to do this)
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save({
            'epoch': (args.train + past_epochs),
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'vocab_size': vocab_size,
            'n_grades': n_grades,
        }, args.model_path)
        print(f"Total epochs over lifetime = {args.train + past_epochs}")
        print(f"\nModel saved to {args.model_path}")

    # get beginning and ending tokens
    bos_token = full_dataset.token_dict.get('BOSr12', 0)
    eos_token = full_dataset.token_dict.get('EOSr14', 1)

    # put these tokens infront of the sequence
    prompt_seq = [bos_token] + args.prompt

    # make seq a tensor
    prompt_seq_tensor = torch.tensor(prompt_seq, dtype=torch.long).unsqueeze(0).to(device)
    # turn grade to tensor
    grade_tensor = torch.tensor([args.grade], dtype=torch.float).to(device)

    with torch.no_grad():

        # temporarily commented out to test decoder on its own
        # result, encoded = encoder(prompt_seq_tensor)
        # mem = encoded.transpose(0, 1).contiguous()

        generated = decoder.generate(
            # memory=mem,
            grade=args.grade,
            max_len=24,
            bos_token=bos_token,
            eos_token=eos_token,
            device=device,
            prompt=prompt_seq_tensor,
        )

    gen_list = generated.squeeze(0).cpu().tolist()
    # make sure the sequence ends in eos_token
    if gen_list[-1] != eos_token:
        gen_list += [eos_token]

    print("the generated token ids for the climb is: ", gen_list)
    try:
        show_climb(gen_list, "data")
    except Exception as e:
        print(f"Could not visualize climb: {e}")


if __name__ == "__main__":
    main()
