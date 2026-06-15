
FINAL_VJEPA2_T8_FusedMean_PowerMean7_q05

Modelo final:
V-JEPA2 T=8 fused mean embedding [1024] -> 7 MLP classifiers -> Power Mean Ensemble q=0.5 -> acción final.

Archivos incluidos:
1. best_OFFICIAL_vjepa2_t8_fused_mean_mlp_plain.pt
2. Exp5A_plain_MLP_h512_drop0.25_lr0.0001_ls0.05.pt
3. Exp6A_mlp_seed11_h512_drop025_lr1e4_ls005.pt
4. Exp6A_mlp_seed22_h512_drop030_lr1e4_ls005.pt
5. Exp6A_mlp_seed33_h512_drop025_lr3e4_ls003.pt
6. Exp6A_mlp_seed44_h256_drop025_lr1e4_ls005.pt
7. Exp6A_mlp_seed55_h512_drop020_lr1e4_ls003.pt

Entrada esperada:
z_fused con shape [1024] o [B, 1024]

Mejor resultado:
Accuracy = 0.878141
F1 macro = 0.860986
F1 weighted = 0.874309
