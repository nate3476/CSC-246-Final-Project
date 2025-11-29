import matplotlib.pyplot as plt
import torch
import numpy as np
from climb_data import KilterDataset
from decoder import ClimbDecoder
from climb_mlp import NeuralNet
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--decoder_path", type=str, default="models/decoder.pt")
    parser.add_argument("--mlp_path", type=str, default="models/mlp_grader.pt")
    parser.add_argument("--N", type=int, default=500)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up the transformer
    full_dataset = KilterDataset(root='data', download=True)
    vocab_size = len(full_dataset.token_dict)
    n_grades = 39
    decoder = ClimbDecoder(vocab_size=vocab_size, n_grades=n_grades).to(device)

    # load the stored decoder model
    print(f"Loading model from {args.decoder_path}")
    loaded_model = torch.load(args.decoder_path, map_location=device)
    decoder.load_state_dict(loaded_model['model_state_dict'])
    print(f"Loaded model from epoch {loaded_model.get('epoch', 'unknown')}")

    # set up and load the mlp for classifying grades
    mlp = NeuralNet(root='data').to(device)
    checkpoint = torch.load(args.mlp_path, map_location=device)
    mlp.load_state_dict(checkpoint["model_state_dict"])
    mlp.eval()

    bos_token = full_dataset.token_dict.get('BOSr12', 0)
    eos_token = full_dataset.token_dict.get('EOSr14', 1)
    prompt_seq_tensor = torch.tensor([bos_token], dtype=torch.long).unsqueeze(0).to(device)

    grade_range = range(12, 29)
    mse = {grade: 0 for grade in grade_range}

    for grade in grade_range:
        for _ in range(args.N):
            generated = decoder.generate(
                grade=grade,
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

            mlp_grade = mlp(generated).item()
            mse[grade] += (grade - mlp_grade) ** 2

        mse[grade] /= args.N
        print(f"MSE at grade {grade}: {mse[grade]}")

    plt.bar(mse.keys(), [np.sqrt(e) for e in mse.values()])
    plt.xlabel("Difficulty")
    plt.ylabel("Root MSE")
    plt.title("Root MSE from prompted difficulty")

    plt.show()


if __name__ == "__main__":
    main()
