Forked from Microsoft's Differential Transformer

https://github.com/microsoft/unilm/tree/master/Diff-Transformer

# Introspective Attention:
by Eric Hartford

For input $X \in \mathbb{R}^{n \times d}$ and $i \in \{1,2,3\}$:

$P_i = \{Q_i, K_i, V_i, A_i\}$

where:

$A_i = \text{LayerNorm}(\text{softmax}(Q_iK_i^T/\sqrt{d_k})V_i)$

$\tilde{K_i},\tilde{V_i} = \text{concat}(P_0...P_{i-1})$ &nbsp; &nbsp; # dims: $\mathbb{R}^{n \times (i(d_k + d_v))}$

$\lambda_i = \text{LayerNorm}(\text{sigmoid}(w_{\lambda_i}))$ # Stabilized weights

$\text{Output} = \text{LayerNorm}(\sum(\lambda_iA_i) + \alpha X)$ &nbsp; &nbsp; # Residual connection

**Key properties:**
1. Path[$i$] has parallel access to Path[0...$i$-1]
2. Dimensionality preserved through projections
3. Gradient and scale stabilized

![image](https://github.com/user-attachments/assets/2e39f22d-5bc0-4339-be4d-2eff26fdcd1e)

![image](https://github.com/user-attachments/assets/cdc103b7-6e78-49c0-8364-2a2f61b4e8cd)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
