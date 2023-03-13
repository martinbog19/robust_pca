# robust_pca

This repo explores the use of robust principal components analysis (RCPA) to isolate corrupted data in images.
The alternating directions method (ADM) is used.
Noise (salt-pepper & normally distributed) is added to common image datasets.![RPCA_faces_hotcmap](https://user-images.githubusercontent.com/104741563/224803077-00602789-3641-4453-9f5b-6283719a09c0.jpeg)

RCPA decomposes the noisy matrix X into a low-rank matrix L and a sparse matrix S such that X = L + S.




https://doi.org/10.1017/9781009089517 (Brunton & Kutz, 2022)



![RPCA_faces_hotcmap](https://user-images.githubusercontent.com/104741563/224803077-00602789-3641-4453-9f5b-6283719a09c0.jpeg)




