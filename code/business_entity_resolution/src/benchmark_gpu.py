"""
Benchmark Sparse TF-IDF Top-K and Sentence Transformer GPU Speed
"""
import time
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

print("Checking PyTorch GPU ...")
x = torch.randn(1000, 384, device='cuda')
y = torch.randn(50000, 384, device='cuda')
t0 = time.time()
sim = torch.mm(x, y.T)
torch.cuda.synchronize()
print(f"GPU matrix multiplication (1000 x 50000): {time.time() - t0:.4f}s")

print("\nChecking Sparse TF-IDF Top-N ...")
names = [f"business company name {i} inc street address" for i in range(50000)]
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
t0 = time.time()
X = vec.fit_transform(names)
print(f"TFIDF fit_transform 50k names: {time.time() - t0:.2f}s, shape: {X.shape}")

t0 = time.time()
# compute top 20 matches per row
res = sp_matmul_topn(X[:1000], X.T, top_n=20, threshold=0.3)
print(f"Sparse top-N matmul 1000 x 50000: {time.time() - t0:.4f}s")
