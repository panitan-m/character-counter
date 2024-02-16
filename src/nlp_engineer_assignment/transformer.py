import math
import random
import torch
import torch.nn as nn
import numpy as np
from .utils import count_letters, score


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
     
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Embeddings(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size=256,
                 dropout=0.1,
                 max_position_embeddings=20):
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.position_ids = torch.arange(max_position_embeddings).expand((1, -1))
        
    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings += position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        

class SelfAttention(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 num_attention_heads=4,
                 dropout=0.1):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None):
        query = self.split_heads(self.query(hidden_states))
        key = self.split_heads(self.key(hidden_states))
        value = self.split_heads(self.value(hidden_states))
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention masks
        if attention_mask is not None:
            attention_scores = attention_mask.masked_fill(attention_mask == 0, -1e9)
            
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        context = torch.matmul(attention_probs, value)
        
        # Combine heads
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(new_context_shape)
        
        return context
        
        
class SelfAttentionResidual(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 dropout=0.1):
        super().__init__()
        self.fully_connected = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.fully_connected(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states
    
    
class FeedForward(nn.Module):
    """
    Positon-wise Feed-Forward Networks
    """
    def __init__(self,
                 hidden_size=256,
                 intermediate_size=1024):
        super().__init__()
        self.fully_connected = nn.Linear(hidden_size, intermediate_size)
        
    def forward(self, hidden_states):
        hidden_states = self.fully_connected(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        return hidden_states
    
    
class FinalResidual(nn.Module):
    def __init__(self,
                 intermediate_size=1024,
                 hidden_size=256,
                 dropout=0.1):
        super().__init__()
        self.fully_connected = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.fully_connected(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states
        
        
class EncoderLayer(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 num_attention_heads=4,
                 intermediate_size=1024,
                 dropout=0.1):
        super().__init__()
        
        self.self_attention = SelfAttention(hidden_size, num_attention_heads, dropout)
        self.self_attention_residual = SelfAttentionResidual(hidden_size, dropout)
        self.feedforward = FeedForward(hidden_size, intermediate_size)
        self.final_residual = FinalResidual(intermediate_size, hidden_size, dropout) 
            
    def forward(self, hidden_states, attention_mask):
        
        attention_output = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.self_attention_residual(attention_output, hidden_states)
        feedforward_output = self.feedforward(hidden_states)
        hidden_states = self.final_residual(feedforward_output, hidden_states)
        return hidden_states
        
        
class Encoder(nn.Module):
    """
    Multi-layer Encoder
    """
    def __init__(self,
                 hidden_size=256,
                 num_attention_heads=4,
                 intermediate_size=1024,
                 num_layer=2,
                 dropout=0.1):
        super().__init__()
        self.layer = nn.ModuleList([EncoderLayer(hidden_size, num_attention_heads, intermediate_size, dropout) for _ in range(num_layer)])
        
    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
    

class Transformer(nn.Module):
    """
    TODO: You should implement the Transformer model from scratch here. You can
    use elementary PyTorch layers such as: nn.Linear, nn.Embedding, nn.ReLU, nn.Softmax, etc.
    DO NOT use pre-implemented layers for the architecture, positional encoding or self-attention,
    such as nn.TransformerEncoderLayer, nn.TransformerEncoder, nn.MultiheadAttention, etc.
    """
    def __init__(self, 
                 vocab_size,
                 sequence_length=20,
                 hidden_size=256, 
                 num_layer=2,
                 num_attention_heads=4,
                 intermediate_size=1024,
                 dropout=0.1,
                 use_attention_mask=True,
                 num_labels=3):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, hidden_size, dropout, sequence_length)
        self.encoder = Encoder(hidden_size, num_attention_heads, intermediate_size, num_layer, dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        if use_attention_mask:
            self.attention_mask = (1 - torch.triu(torch.ones(1, sequence_length, sequence_length), diagonal=1))
        else:
            self.attention_mask = None
        
    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        outputs = self.encoder(hidden_states, self.attention_mask)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        predictions = torch.argmax(logits, dim=-1)
        return logits, predictions
    
    
def generate_batches(data, batch_size, char_to_index):
    data_length = len(data)
    indices = list(range(data_length))
    random.shuffle(indices)
    
    for i in range(0, data_length, batch_size):
        batch_indices = indices[i:i + batch_size]
        
        # String batch
        strings = [data[j] for j in batch_indices]
        
        # Generate a batch of input ids from string batch
        input_ids = torch.tensor(list(map(lambda x: [char_to_index[char] for char in x], strings)))
        
        # Generate a batch of labels from string batch
        labels = torch.tensor(np.array(list(map(count_letters, strings)))).long()
        
        yield (input_ids, labels)

        
def train_classifier(train_inputs, vocabs):
    # TODO: Implement the training loop for the Transformer model.
    
    # Character to index dictionary
    char_to_index = {char: index for index, char in enumerate(vocabs)}
    
    batch_size = 16
    
    num_train_inputs = len(train_inputs)
    num_batches = num_train_inputs // batch_size
    
    vocab_size = len(vocabs)
    sequence_length = 20
    hidden_size = 256 
    num_layer = 2
    num_attention_heads = 4
    intermediate_size = 1024
    dropout = 0.2
    use_attention_mask = True
    num_labels = 3
                     
    model = Transformer(
        vocab_size,
        sequence_length,
        hidden_size,
        num_layer,
        num_attention_heads,
        intermediate_size,
        dropout,
        use_attention_mask,
        num_labels)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    total_loss = 0
    total_accuracy = 0
    log_interval = 100
    
    
    epochs = 10
    for epoch in range(epochs):
        for i, batch in enumerate(generate_batches(train_inputs, batch_size, char_to_index)):
            input_ids, targets = batch
            logits, predictions = model(input_ids)
            
            num_labels = logits.size()[-1]
            loss = criterion(logits.view(-1, num_labels), targets.view(-1))
            accuracy = score(targets.numpy(), predictions.numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += accuracy
            if i % log_interval == 0 and i > 0:
                current_loss = total_loss / log_interval
                current_accuracy = total_accuracy / log_interval
                print(f'| Epoch {epoch+1:3d} | {i:3d}/{num_batches:3d} Batches | Loss {current_loss:3.2f} | Accuracy {current_accuracy:3.2f} |')
                total_loss = 0
                total_accuracy = 0
                
        total_loss = 0
        total_accuracy = 0
                
    return model
