
import numpy as np
import torch.nn as nn
import torch
import gc
import torch.nn.functional as F
import math

class CueBasedRNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers,
                 embedding_file=None, dropout=0.5, tie_weights=False, freeze_embedding=False,
                 aux_objective=False, nauxclasses=0, ablate_attention=False, uniform_attention=False, device=None):
        super().__init__()
        self.device = device
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.score_attn = nn.Softmax(dim=-1)

        if embedding_file:
            # Use pre-trained embeddingss
            embed_weights = self.load_embeddings(embedding_file, ntoken, ninp)
            self.encoder = nn.Embedding.from_pretrained(embed_weights)
        else:
            self.encoder = nn.Embedding(ntoken+1, ninp)
        
        #generate query from hidden state and embedding
        self.q = nn.Linear(ninp+nhid,nhid)
        #project from prev hidden state, embedding, query, attn to large intermediate layer
        self.ablate_attention = ablate_attention
        self.uniform_attention = uniform_attention
        if(ablate_attention):
            self.intermediate_h = nn.Linear(nhid*3,nhid*4)
            self.final_h = nn.Linear(nhid*4,nhid)
        else:
            self.intermediate_h = nn.Linear(nhid*4,nhid*4)
            #from large intermediate layer to current word key, value, and next-word prediction
            self.final_h = nn.Linear(nhid*4,nhid*3)

        self.decoder = nn.Linear(nhid, ntoken+1)
        self.aux_objective = aux_objective
        if(aux_objective):
            self.aux_decoder = nn.Linear(nhid, nauxclasses)

        self.init_weights(freeze_embedding, aux_objective=False)
        if freeze_embedding:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        if(self.ablate_attention):
            self.f_norm = torch.nn.LayerNorm(nhid)
        else:
            self.f_norm = torch.nn.LayerNorm(nhid * 3)            

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.nhid = nhid
        self.attn_div_factor = np.sqrt(nhid)

    def init_weights(self, freeze_embedding, aux_objective):
        """ Initialize encoder and decoder weights """
        initrange = 0.1
        if not freeze_embedding:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if(aux_objective):
            self.aux_decoder.bias.data.fill_(0)
            self.aux_decoder.weight.data.uniform_(-initrange, initrange)

    def zero_parameters(self):
        """ Set all parameters to zero (likely as a baseline) """
        self.encoder.weight.data.fill_(0)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.fill_(0)
        for weight in self.rnn.parameters():
            weight.data.fill_(0)

    def random_parameters(self):
        """ Randomly initialize all RNN parameters but not the encoder or decoder """
        initrange = 0.1
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.uniform_(-initrange, initrange)

    def load_embeddings(self, embedding_file, ntoken, ninp):
        """ Load pre-trained embedding weights """
        weights = np.empty((ntoken, ninp))
        with open(embedding_file, 'r') as in_file:
            ctr = 0
            for line in in_file:
                weights[ctr, :] = np.array([float(w) for w in line.strip().split()[1:]])
                ctr += 1
        return(torch.tensor(weights).float())
    
    def add_attention_head(self):
        """ Adds an extra attention head by expanding the key and value projections. """
        self.nheads += 1
        nhid = self.nhid
        
        # Expand the existing layers to accommodate the new attention head
        new_final_h = nn.Linear(nhid * 4, nhid * (2 + self.nheads))
        new_final_h.weight.data[:self.final_h.out_features, :] = self.final_h.weight.data
        new_final_h.bias.data[:self.final_h.out_features] = self.final_h.bias.data
        
        self.final_h = new_final_h.to(self.device)
        print(f"Added a new attention head. Total heads: {self.nheads}")
    
    def update_attention_heads(self, epoch, n):
        """ Adds an attention head every n epochs """
        if epoch % n == 0:
            self.add_attention_head()
    #b = batch size, n = sequence length, d = dimensionality 
    #masks are of size b * n * n+1 - dim 3 is token a attending token b in dim 4
    def forward(self, observation, initial_cache, masks=None, attn_softmax_scaling_factor=1, output_attn=False, uniform_attn=False, random_attn=False):
        #todo - initialize outside of forward pass
        hidden, key_cache, value_cache = initial_cache
        seq_len = observation.size(dim=0)
        emb = self.drop(self.encoder(observation))
        if(output_attn):
            attn_log = {'weights':[],'scores':[]}
        else:
            attn_log = None
        for i in range(seq_len):
            #self-attention
            #generate query from prev. hidden state and curr. word embedding
            query = self.drop(self.tanh(self.q_norm(self.q(torch.cat((emb[i],hidden[i]), -1))))) #b * d
            query_n = query.unsqueeze(-1) #b * n * 1
            if(not self.ablate_attention):
                if(self.uniform_attention or uniform_attn):
                    attn_scores = torch.zeros(masks[:,i,:i+1].shape).to(self.device)
                elif(random_attn):
                    attn_scores = torch.rand(masks[:,i,:i+1].shape).to(self.device)
                else:
                    attn_scores = torch.bmm(key_cache.swapaxes(0,1), query_n).squeeze(dim=-1)
                if(masks is not None):
                    if attn_scores.shape[0]!=masks.shape[0]:
                        masks = masks[:attn_scores.shape[0], ...]
                    masked_scores = attn_scores + masks[:,i,:i+1]
                else:
                    masked_scores = attn_scores
                #divide scores by sqrt(nhid) for more stable gradients, then compute score using specified function (default: softmax)
                masked_scores = masked_scores * (1 / self.attn_div_factor)
                attn_weights = self.score_attn(masked_scores * attn_softmax_scaling_factor)
                if(output_attn):
                    attn_log['weights'].append(attn_weights)
                    attn_log['scores'].append(masked_scores)
                attn = (attn_weights.T.unsqueeze(-1) * value_cache).sum(axis=0)

                #feed-forward component
                #project to large intermediate layer
                intermediate = self.drop(self.tanh(self.int_norm(self.intermediate_h(torch.cat((emb[i],query,attn,hidden[i]),-1)))))
                #project to final layer to generate current word key, final hidden state used for prediction
                key_cache_i, value_cache_i, hidden_i = self.drop(self.tanh(self.f_norm(self.final_h(intermediate)))).split(self.nhid, dim=-1)
                #update memory cache for attention and hidden states. Currently inefficent
                #Can be changed by intializing a tnesor of zeros of dim (seq_len, batch_size, nhid) before the loop and just appending on it 
                hidden = torch.cat((hidden, hidden_i.unsqueeze(0)), dim=0)
                key_cache = torch.cat((key_cache, key_cache_i.unsqueeze(0)), dim=0)
                value_cache = torch.cat((value_cache, value_cache_i.unsqueeze(0)), dim=0)
            else:
                intermediate = self.drop(self.tanh(self.int_norm(self.intermediate_h(torch.cat((emb[i],query,hidden[i]),-1)))))
                hidden_i = self.drop(self.tanh(self.f_norm(self.final_h(intermediate))))
                hidden = torch.cat((hidden, hidden_i.unsqueeze(0)), dim=0)
            #delete temporary variables
            del query, query_n, attn_scores, masked_scores, attn_weights, attn, intermediate, key_cache_i, value_cache_i, hidden_i
            gc.collect() 
        output = hidden[1:]
        decoded = self.decoder(output)
        if(self.aux_objective):
            decoded_aux = self.aux_decoder(output)
        else:
            decoded_aux = None

        return decoded, hidden, decoded_aux, attn_log

    def init_cache(self, observation):
        device = observation.device
        if len(observation.size())>1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1
        seq_len = observation.size(dim=0)

        return torch.zeros(1, bsz, self.nhid, device=device), torch.zeros(1, bsz, self.nhid, device=device), torch.zeros(1, bsz, self.nhid, device=device)

    def set_parameters(self,init_val):
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.fill_(init_val)
            self.encoder.weight.data.fill_(init_val)
            self.decoder.weight.data.fill_(init_val)

    def randomize_parameters(self):
        initrange = 0.1
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.uniform_(-initrange, initrange)

class OptimizedCueBasedRNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken+1, ninp)       
        self.q = nn.Linear(ninp+nhid, nhid)
        self.intermediate_h = nn.Linear(nhid*4, nhid*4)
        self.final_h = nn.Linear(nhid*4, nhid*3)
        self.decoder = nn.Linear(nhid, ntoken+1)
        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        self.f_norm = torch.nn.LayerNorm(nhid * 3)            
        self.nhid = nhid
        self.attn_div_factor = np.sqrt(nhid)
        
        self.init_weights()

    def init_weights(self):
        """ Initialize encoder and decoder weights """
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def random_parameters(self):
        """ Randomly initialize all RNN parameters but not the encoder or decoder """
        initrange = 0.1
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.uniform_(-initrange, initrange)

    def create_causal_mask_vectorized(self, seq_len, device):
        """Create causal mask for all positions at once"""
        # Create a upper triangular mask for all positions
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, observation, initial_cache):
        hidden_init, key_cache_init, value_cache_init = initial_cache
        seq_len, batch_size = observation.shape
        
        # Pre-allocate tensors to avoid repeated concatenations
        device = observation.device
        hidden_states = torch.zeros(seq_len + 1, batch_size, self.nhid, device=device)
        key_cache_states = torch.zeros(seq_len + 1, batch_size, self.nhid, device=device)  
        value_cache_states = torch.zeros(seq_len + 1, batch_size, self.nhid, device=device)
        
        # Initialize with provided cache
        hidden_states[0] = hidden_init.squeeze(0)
        key_cache_states[0] = key_cache_init.squeeze(0)
        value_cache_states[0] = value_cache_init.squeeze(0)
        
        # Embed all tokens at once
        emb = self.drop(self.encoder(observation))  # seq_len x batch_size x ninp
        
        # Create causal mask once for all positions
        causal_mask = self.create_causal_mask_vectorized(seq_len, device)
        
        for i in range(seq_len):
            # Current cache length (including current position)
            cache_len = i + 1
            
            # Get current embeddings and hidden state
            curr_emb = emb[i]  # batch_size x ninp
            curr_hidden = hidden_states[i]  # batch_size x nhid
            
            # Query computation
            query_input = torch.cat([curr_emb, curr_hidden], dim=-1)
            query = self.drop(self.tanh(self.q_norm(self.q(query_input))))  # batch_size x nhid
            
            # Attention computation - vectorized over cache
            current_keys = key_cache_states[:cache_len]  # cache_len x batch_size x nhid
            current_values = value_cache_states[:cache_len]  # cache_len x batch_size x nhid
            
            # Compute attention scores: (cache_len x batch_size x nhid) @ (batch_size x nhid x 1)
            attn_scores = torch.bmm(
                current_keys.transpose(0, 1),  # batch_size x cache_len x nhid
                query.unsqueeze(-1)  # batch_size x nhid x 1
            ).squeeze(-1)  # batch_size x cache_len
            
            # Apply causal mask and scaling
            masked_scores = attn_scores + causal_mask[i, :cache_len].unsqueeze(0)
            masked_scores = masked_scores / self.attn_div_factor
            attn_weights = F.softmax(masked_scores, dim=-1)  # batch_size x cache_len
            
            # Compute attention output
            attn = torch.bmm(
                attn_weights.unsqueeze(1),  # batch_size x 1 x cache_len
                current_values.transpose(0, 1)  # batch_size x cache_len x nhid  
            ).squeeze(1)  # batch_size x nhid
            
            # Intermediate computation
            intermediate_input = torch.cat([curr_emb, query, attn, curr_hidden], dim=-1)
            intermediate = self.drop(self.tanh(self.int_norm(self.intermediate_h(intermediate_input))))
            
            # Final computation and splitting
            final_output = self.drop(self.tanh(self.f_norm(self.final_h(intermediate))))
            key_i, value_i, hidden_i = final_output.split(self.nhid, dim=-1)
            
            # Store in pre-allocated tensors
            hidden_states[i + 1] = hidden_i
            key_cache_states[i + 1] = key_i
            value_cache_states[i + 1] = value_i
            
        # Output processing
        output = hidden_states[1:]  # Remove initial state
        decoded = self.decoder(output)
        
        return decoded, hidden_states

    def init_cache(self, observation):
        device = observation.device
        if len(observation.size()) > 1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1

        return (torch.zeros(1, bsz, self.nhid, device=device), 
                torch.zeros(1, bsz, self.nhid, device=device), 
                torch.zeros(1, bsz, self.nhid, device=device))

    def set_parameters(self, init_val):
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.fill_(init_val)
        self.encoder.weight.data.fill_(init_val)
        self.decoder.weight.data.fill_(init_val)

    def randomize_parameters(self):
        initrange = 0.1
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.uniform_(-initrange, initrange)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OptimizedCueBasedRNNModel_MH(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, nheads, dropout=0.5):
        super().__init__()
        
        assert nhid % nheads == 0, f"nhid ({nhid}) must be divisible by nheads ({nheads})"

        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken + 1, ninp)
        
        # Query projection for current state
        self.q = nn.Linear(ninp + nhid, nhid)
        
        # Multi-head attention projections - separate for each head
        self.nheads = nheads
        self.d_k = nhid // nheads
        self.attn_scale = 1.0 / math.sqrt(self.d_k)
        
        self.query_projections = nn.ModuleList([
            nn.Linear(nhid, self.d_k, bias=False) for _ in range(nheads)
        ])
        self.key_projections = nn.ModuleList([
            nn.Linear(nhid, self.d_k, bias=False) for _ in range(nheads)
        ])
        self.value_projections = nn.ModuleList([
            nn.Linear(nhid, self.d_k, bias=False) for _ in range(nheads)
        ])
        
        # Output projection for multi-head attention
        self.output_projection = nn.Linear(nhid, nhid, bias=False)
        
        # RNN components
        self.intermediate_h = nn.Linear(nhid * 4, nhid * 4)  # emb + query + attn + hidden
        self.final_h = nn.Linear(nhid * 4, nhid * 3)  # outputs: key, value, hidden
        
        # Decoder
        self.decoder = nn.Linear(nhid, ntoken + 1)
        
        # Layer normalization
        self.q_norm = nn.LayerNorm(nhid)
        self.int_norm = nn.LayerNorm(nhid * 4)
        self.f_norm = nn.LayerNorm(nhid * 3)
        
        self.nhid = nhid
        self.ninp = ninp
        
        self.init_weights()

    def init_weights(self):
        """Initialize encoder and decoder weights"""
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
        # Initialize attention projections
        for module in self.query_projections + self.key_projections + self.value_projections:
            nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def multi_head_attention(self, query, keys, values):
        """
        Multi-head attention computation for sequential processing
        
        Args:
            query: [batch_size, nhid] - current query state
            keys: [cache_len, batch_size, nhid] - all previous keys including current
            values: [cache_len, batch_size, nhid] - all previous values including current
        
        Returns:
            attended: [batch_size, nhid] - attended representation
            attn_weights: [batch_size, nheads, cache_len] - attention weights per head
        """
        batch_size = query.shape[0]
        cache_len = keys.shape[0]
        
        all_attention_weights = []
        head_outputs = []
        
        # Process each head separately
        for i in range(self.nheads):
            # Project query for this head
            q_i = self.query_projections[i](query)  # [batch_size, d_k]
            
            # Project all keys and values for this head
            k_i = self.key_projections[i](keys)     # [cache_len, batch_size, d_k]
            v_i = self.value_projections[i](values) # [cache_len, batch_size, d_k]
            
            # Compute attention scores: query vs all keys
            # q_i: [batch_size, d_k], k_i: [cache_len, batch_size, d_k]
            scores = torch.einsum('bd,tbd->bt', q_i, k_i) * self.attn_scale
            # scores: [batch_size, cache_len]
            
            if cache_len > 1:
                # Mask out the last position (current position i) so we only attend to previous positions
                scores[:, -1] = float('-inf')  # Mask the current position
            
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(scores, dim=-1)  # [batch_size, cache_len]
            attention_weights = self.drop(attention_weights)
            
            # Apply attention to values
            # attention_weights: [batch_size, cache_len], v_i: [cache_len, batch_size, d_k]
            head_output = torch.einsum('bt,tbd->bd', attention_weights, v_i)
            # head_output: [batch_size, d_k]
            
            head_outputs.append(head_output)
            all_attention_weights.append(attention_weights)
        
        # Concatenate all head outputs
        concatenated = torch.cat(head_outputs, dim=-1)  # [batch_size, nhid]
        
        # Final linear projection
        attended = self.output_projection(concatenated)  # [batch_size, nhid]
        
        # Stack attention weights for analysis
        attn_weights = torch.stack(all_attention_weights, dim=1)  # [batch_size, nheads, cache_len]
        
        return attended, attn_weights
    
    def forward(self, observation, initial_cache):
        """
        Sequential forward pass maintaining CBR-RNN structure
        """
        hidden_init, key_cache_init, value_cache_init = initial_cache
        seq_len, batch_size = observation.shape
        device = observation.device
        
        # Pre-allocate tensors
        hidden_states = torch.zeros(seq_len + 1, batch_size, self.nhid, device=device)
        key_cache_states = torch.zeros(seq_len + 1, batch_size, self.nhid, device=device)
        value_cache_states = torch.zeros(seq_len + 1, batch_size, self.nhid, device=device)
        
        # Initialize with provided cache
        hidden_states[0] = hidden_init.squeeze(0)
        key_cache_states[0] = key_cache_init.squeeze(0)
        value_cache_states[0] = value_cache_init.squeeze(0)
        
        # Embed all tokens at once
        emb = self.drop(self.encoder(observation))  # [seq_len, batch_size, ninp]
        
        # Sequential processing - this is the key CBR-RNN characteristic
        for i in range(seq_len):
            # Current cache length (including current position after we add to cache)
            cache_len = i + 1
            
            # Get current embeddings and previous hidden state
            curr_emb = emb[i]  # [batch_size, ninp]
            prev_hidden = hidden_states[i]  # [batch_size, nhid]
            
            # 1. QUERY: Compute query from current input and previous hidden state
            query_input = torch.cat([curr_emb, prev_hidden], dim=-1)
            query = self.drop(self.tanh(self.q_norm(self.q(query_input))))  # [batch_size, nhid]
            
            # 2. UPDATE CACHE: Add current query as key/value to cache
            key_cache_states[cache_len - 1] = query  # Use query as both key and value initially
            value_cache_states[cache_len - 1] = query
            
            # 3. ATTENTION: Attend to all previous states (including current)
            current_keys = key_cache_states[:cache_len]    # [cache_len, batch_size, nhid]
            current_values = value_cache_states[:cache_len] # [cache_len, batch_size, nhid]
            
            attn, attn_weights = self.multi_head_attention(query, current_keys, current_values)
            
            # 4. INTERMEDIATE: Combine all information
            intermediate_input = torch.cat([curr_emb, query, attn, prev_hidden], dim=-1)
            intermediate = self.drop(self.tanh(self.int_norm(self.intermediate_h(intermediate_input))))
            
            # 5. FINAL: Compute final outputs
            final_output = self.drop(self.tanh(self.f_norm(self.final_h(intermediate))))
            key_final, value_final, hidden_final = final_output.split(self.nhid, dim=-1)
            
            # 6. UPDATE: Store final states
            hidden_states[i + 1] = hidden_final
            # Update cache with refined key/value representations
            key_cache_states[i + 1] = key_final
            value_cache_states[i + 1] = value_final
        
        # Output processing (exclude initial hidden state)
        output = hidden_states[1:]  # [seq_len, batch_size, nhid]
        decoded = self.decoder(output)  # [seq_len, batch_size, ntoken + 1]
        
        # Return final cache states
        final_cache = (
            hidden_states[-1:],      # Last hidden state
            key_cache_states[-1:],   # Last key state  
            value_cache_states[-1:]  # Last value state
        )
        
        return decoded, final_cache

    def init_cache(self, observation):
        """Initialize empty cache"""
        device = observation.device
        if len(observation.size()) > 1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1

        return (torch.zeros(1, bsz, self.nhid, device=device), 
                torch.zeros(1, bsz, self.nhid, device=device), 
                torch.zeros(1, bsz, self.nhid, device=device))

    def set_parameters(self, init_val):
        """Set all parameters to a specific value"""
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.fill_(init_val)
        self.encoder.weight.data.fill_(init_val)
        self.decoder.weight.data.fill_(init_val)
        
        # Set attention parameters
        for module_list in [self.query_projections, self.key_projections, self.value_projections]:
            for module in module_list:
                module.weight.data.fill_(init_val)
        self.output_projection.weight.data.fill_(init_val)

    def randomize_parameters(self):
        """Randomly initialize parameters"""
        self.init_weights()
        initrange = 0.1
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.uniform_(-initrange, initrange)