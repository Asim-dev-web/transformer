# Transformer Architectur

## Overview 

This is a single-head attention transformer.

---


## dimensions

vocabulary: 1000 (total tokens known to the model)
batch_size: 32 (number of distinct batches fed during training at once)
block_size: 256 (number of tokens in one batch)
d_model: 256 (number of dimensions each token is divided in)


embedding: 32 x 256 x 256 (batch_size x block_size x d_model)
embedding_pos: embedding + pos
embedding_layer_norm = layernorm(embedding_pos)

attention_weights (q_weight, k_weight, v_weights): 256 x 256 (d_model x d_model/no. of attention)

Q, K, V (embedding @ attention_weights): 32 x 256 x 256 (batch_size x block_size x d_model/no. of attention)

A = softmax(Q @ K.T/underroot(d_model)): 32 x 256 x 256 (batch_size x block_size x block_size)
A_output = A @ V: 32 X 256 X 256 (batch_size x block_size x d_model/no. of attention)

A_residual = A_output + embedding_pos
A_layer_norm = layernorm(A_residual)