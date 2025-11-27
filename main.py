import torch
from torch.utils.data import DataLoader
from climb_data import KilterDataset, climb_collate_fn
from encoder import ClimbEncoder
from decoder import ClimbDecoder
from climb_util import show_climb 
import argparse 

def main():
  
  parser = argparse.ArgumentParser()

  parser.add_argument("--grade", type = float, required = True)
  parser.add_argument("--prompt", type = int, nargs= "*" , default=[])

  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dataset = KilterDataset("data", download = True)

  dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn = climb_collate_fn)

  vocab_size = len(dataset.token_dict)

  n_grades = 11

  encoder = ClimbEncoder(vocab_size).to(device)
  decoder = ClimbDecoder(vocab_size=vocab_size, n_grades = n_grades).to(device)

  #get beginning and ending tokens 
  bos_token = dataset.token_dict.get('BOSr12', 0)
  eos_token = dataset.token_dict.get('EOSr14', 1)

  #put these tokens infront of the sequence
  prompt_seq = [bos_token] + args.prompt

  #make seq a tensor
  prompt_seq_tensor = torch.tensor(prompt_seq,dtype = torch.long).unsqueeze(0).to(device)

  #turn grade to tensor
  grade_tensor = torch.tensor([args.grade], dtype=torch.float).to(device) 

  with torch.no_grad():
  
    result, encoded = encoder(prompt_seq_tensor)
  
    mem = encoded.transpose(0, 1).contiguous()
        
    generated = decoder.generate(
        memory=mem,
        max_len=128,
        bos_token=bos_token,
        eos_token=eos_token,
        device=device
    )
  
  gen_list = generated.squeeze(0).cpu().tolist()

  #TODO: input data root here
  show_climb(gen_list, "data")

  print("the generated token ids for the climb is: ", gen_list)

if __name__ == "__main__":
  main()
