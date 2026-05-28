import numpy as np

class Transformer:
    def __init__(self):
        rng = np.random.default_rng()
        self.vocab = 1000
        self.d_model = 256
        self.batch_size = 32
        self.block_size = 256
        self.embeddings = rng.normal(0.00,0.02,(self.vocab,self.d_model))
        self.w_q = rng.normal(0.00,0.02,(self.d_model,self.d_model))
        self.w_k = rng.normal(0.00,0.02,(self.d_model,self.d_model))
        self.w_v = rng.normal(0.0,0.02,(self.d_model,self.d_model))
        self.gamma = rng.uniform(1,1, (256,))
        self.beta = rng.uniform(0,0, (256,))
        self.gamma2 = rng.uniform(1,1, (256,))
        self.beta2 = rng.uniform(0,0, (256,))
        self.pos = rng.uniform(0,0, (32,256,256))
        self.ffn_w1 = rng.normal(0.00,0.02, (self.d_model,256))
        self.ffn_b1 = rng.uniform(0, 0, (256,))
        self.ffn_w2 = rng.normal(0.00,0.02, (256,256))
        self.ffn_b2 = rng.uniform(0, 0, (256,))
        self.ffn_wO = rng.normal(0.0,0.02, (256,self.vocab))
        
    def layer_norm(self, w, gamma, beta):
        mean = np.mean(w,-1,keepdims=True)
        std = np.std(w,-1,keepdims=True)
        
        w_scaled = (w - mean) / (std + 1e-10)
        return gamma * w_scaled + beta
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
    
    def train(self, x, y, epoches=500):
        
        #data loader step and batch creation, output of this step needs to be a matrix x of shape (32 x 256)
        x_embedding = self.embeddings[x]
        
        # get x_pos by adding x_embedding and x_pos for now this is dummy below
        x_pos = x_embedding + self.pos
        
        x_embedding_layernorm = self.layer_norm(x_pos,self.gamma,self.beta)
        
        Q = x_embedding_layernorm @ self.w_q
        K = x_embedding_layernorm @ self.w_k
        V = x_embedding_layernorm @ self.w_v
        
        A = self.softmax(Q @ np.transpose(K, (0,2,1))/self.d_model**0.5) 
        A_output = A @ V
        
        A_output_residual = A_output + x_pos
        A_output_layernorm = self.layer_norm(A_output_residual, self.gamma2, self.beta2)
        
        ffn_l1 = (A_output_layernorm @ self.ffn_w1) + self.ffn_b1
        ffn_l2 = (ffn_l1 @ self.ffn_w2) + self.ffn_b2
        ffn_o = ffn_l2 @ self.ffn_wO
        
        
        
        
        
    