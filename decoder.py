#Used Isabel's decoder and added a generate feature, I will test and debug 

import torch
import torch.nn as nn
import math # for sqrt

def generate_causal_mask(seq_len, device=None):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) # upper right traingle of square matrix is ones (and diagonal)
    mask = mask.masked_fill(mask == 1, float('-inf')) # set the ones to -infinity (disallowed positions)
    return mask
    # when this mask is added to the attention scores, after softmax this will basically make the attention to future tokens = 0

class ClimbDecoder(nn.Module):

    def __init__(
        self,
        vocab_size, # number of distinct hold tokens
        n_grades, # number of possible grades (12?)
        max_seq_len = 128, # should be changed
        embed_dim=256,
        n_heads=9,
        num_layers=6,
        ff_dim = 512, # feed forward dimension, used in defining a layer
        dropout=0.1,
    ):
        super().__init__()

        ########### Embeddings ###########
        self.token_embed = nn.Embedding(vocab_size, embed_dim) # maps token indices to vectors of length embed_dim
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim) # learnable positional embedding for (up to) max_seq_len positions.
        self.grade_embed = nn.Embedding(n_grades, embed_dim) # maps difficulty tokens to vectors of length embed_dim.

        self.embed_dropout = nn.Dropout(dropout) # to prevent overfitting
        self.embed_scale = math.sqrt(embed_dim) # somehow this stabilizes training, not sure about it. I think it's in "Attention Is All You Need"

        ########### Transformer layers ###########
        # stacks num_layers copies of decoder_layer to make a TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True # size (B, T, embed_dim) is kept throughout
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        ########### Output ###########
        self.norm = nn.LayerNorm(embed_dim) # not sure if this is necessary
        self.fc_out = nn.Linear(embed_dim, vocab_size) # output from hidden states to the vocabulary. One logit per possible hold.
        # this is what we can calculate the loss of

    def forward(self, input_seq, grade_token, pad_mask=None):
        '''
        input_seq: size (B, T) token indices
        grade_token: size (B) difficulties
        pad_mask: size (B, T) boolean mask, true where there's padding. I'm a little unsure about this but saw it in a tutorial
        
        B is batch size, T is sequence length of input tensor

        so input (B, T) of token indices becomes (B, T, embed_dim) of vectors which encode "hold #X at position #Y for grade #G"
        '''
        
        B, T = input_seq.size()
        device = input_seq.device

        ########### Embedding ###########
        positions = torch.arange(0, T, device=input_seq.device).unsqueeze(0) # size (1, T)
        
        # combine the embeddings
        # note because of broadcasting rules, when we add something with "1" as a dimension, it's copied until the size matches
        # i.e. if we add something of size (1, T, embed_dim) to something (B, T, embed_dim), it's as if we added B copies of the (1, T, embed_dim) so every batch is affected
        x = self.token_embed(input_seq) * self.embed_scale # don't fully understand the embed_scale yet. adding size (B, T, embed_dim)
        x += self.pos_embed(positions) # adding size (1, T, embed_dim), since we're learning position importance across batches
        x+= self.grade_embed(grade_token).unsqueeze(1) # adding size (B, 1, embed_dim), since there's one grade per batch
        x = self.embed_dropout(x) # incorporate dropout

        ########### Masking ###########
        causal_mask = generate_causal_mask(T, device=device) # means token t can't see future tokens
        # pad_mask is true where padded so the transformer can ignore padding, need to add this
        # size is (B, T)

        ########### Decode ###########
        out = self.transformer(
            tgt=x,
            memory=None,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )

        ########### Output ###########
        out = self.norm(out) # is this necessary?
        logits = self.fc_out(out) # size (B, T, vocab_size)
        return logits

    def generate(self, memory, max_len=128, bos_token=0, eos_token=1, device=None):
        """
        inputs are:
        memory: (B, S, embed_dim) encoder output
        max_len: maximum sequence length to generate
        bos_token: start-of-sequence token
        eos_token: end-of-sequence token
    ]  output is: 
        generated_seq: (B, T_gen) generated token indices
        """

        mem = memory.size(0)

        gen_seq = torch.full((mem, 1), bos_token, dtype=torch.long, device=device)

        for i in range(max_len):

        logits = self.forward(gen_seq, memory)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  

        gen_seq = torch.cat([gen_seq, next_token], dim=1)

        if (next_token == eos_token).all():
        
            break

        return gen_seq
 
