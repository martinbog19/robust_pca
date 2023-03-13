# robust_pca

This repo explores the use of robust principal components analysis (RCPA) to isolate corrupted data in images. 
The alternating directions method (ADM) is used.
Noise (salt-pepper & normally distributed) is added to common image datasets.
RCPA decomposes the noisy matrix X into a low-rank matrix L and a sparse matrix S such that X = L + S.
