
import torch
from torch import nn
from transformers import AutoTokenizer
from torchview import draw_graph

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# take out the attention mechanism in this other place, and splice. 
class TextSimilarityBlock(nn.Module):
    def __init__(self, embed_dim=1000):
        super(TextSimilarityBlock, self).__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=10)
        self.ffn= nn.Linear(embed_dim, embed_dim) 
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.similarity_head = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, sequence1: torch.Tensor, sequence2: torch.Tensor):
        # Figure out batching later.
        # Can also use self-attention after concatenating seq1 and seq2. 
        sequence1 = self.embedding(sequence1)
        sequence2 = self.embedding(sequence2)

        query = self.query_linear(sequence1)
        key, value = self.key_linear(sequence2), self.value_linear(sequence2)
        attention_output, _ = self.attention(query, key, value, need_weights=False)
        attention_output = self.layer_norm1(attention_output)
        ffn_output = self.ffn(attention_output)
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.layer_norm2(ffn_output)
        similarity = self.sigmoid(self.similarity_head(ffn_output[-1]))
        return similarity
    

"""
seqeunce1 = torch.tensor(tokenizer.encode(sequence1))
seqeunce2 = torch.tensor(tokenizer.encode(sequence2))

"""
model = TextSimilarityBlock()
model.eval()    
dummy_input1 = torch.randint(0, tokenizer.vocab_size, (10, 1))  # Sequence 1 of length 10
dummy_input2 = torch.randint(0, tokenizer.vocab_size, (10, 1))  # Sequence 2 of length 10s

model_graph = draw_graph(model, (dummy_input1, dummy_input2),  save_graph=True)

torch.onnx.export(
    model,                                # Model instance
    (dummy_input1, dummy_input2),         # Input tensors
    "text_similarity_model.onnx",         # Path to save the ONNX model
    input_names=["sequence1", "sequence2"],  # Names for the input layers
    output_names=["similarity"],           # Name for the output layer
    dynamic_axes={
        "sequence1": {0: "batch_size", 1: "sequence_length"},
        "sequence2": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=14  # Ensure compatibility with the environment
)