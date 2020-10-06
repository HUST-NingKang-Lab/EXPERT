# Transferred-ONN4MST --- 起别的名字

## Abstract

- The significant correlation between taxonomic structures and microbiome sources makes possible for accurate microbiome source tracking.
- We have proposed ONN4MST in previews work. It has show highly priori accuracy comparing with other methods. But it still faces several limitations (preference to well-studied microbiome source and weak flexibility) when applied to real world problems.
- Today, relying on advanced transfer learning techniques. We now propose Transferred-ONN4MST.
- It has been a great success on two fronts: Great flexibility like unsupervised learning methods (FEAST, JSD, SourceTracker) and excellent accuracy even higher than ONN4MST. Thus it can provide better solutions under some special circumstances.

## Main

Highlights:

- great flexibility relying on transfer learning
- customizable ontology and phylogenetic tree
- less data needed when retraining
- great efficiency and low memory usage
- higher accuracy on small microbiome

## Results

- The end-to-end ontology-aware Neural Network.
	- an extra feature mapper block
	- feature extraction using NN preprocessing block
	- 4 conceptual blocks
	- Dropout, Batch Normalization
	- post-processing block
- Illustration of source tracking process using Transfer-ONN4MST.
- From easy case to difficult cases (Transfer general model to ...)
	- Human subset, Water subset, Soil subset (**easy tasks**: more specified, but not detailed).
	- Contamination detection, Infant colonization (just so so tasks: more specified and detailed).
	- **More tasks.**
	- Distinguishing ICU patients from heathy adults, Disease detection (hard task: absolutely different ontology structure with details).

## Discussion



## Materials and Methods

- Datasets:

- Methods to be compared:

	- ONN4MST, FEAST, SourceTracker
- Implementation
	- Adam optimizer, batch size, training process (early stopping)
	- transfer process
- Uncertainty weighted loss
- Adaptive dropout rate for each layer,  to reduce overfitting

$$
n_{units}^{\{drouout\}}=n_{units}^{\{current\}} - C * n_{units}^{\{last\}}\\
so,\quad rate^{\{dropout\}} = \frac{n_{units}^{\{drouout\}}}{n_{units}^{\{current\}}} = \frac{n_{units}^{\{current\}} - C * n_{units}^{\{last\}}}{n_{units}^{\{current\}}}
$$

- Feature mapping (from original feature space to PC space) using NN architecture.

	$$
	F = [f_1, f_2, f_3, ..., f_{num_f}]\\
	\hat{F} = BF
	$$
	
- Data augmentation

	- Randomly generate contributions $C_{sample}=[c_1, c_2, c_3, ..., c_m]$ , $m=num_{sources}$
	- Randomly select samples $S=[s_1, s_2, s_, ...s_m]^T, m=num_{sources}$ from source environments
	- Repeat $n=num_{sample}$ times, to form a matrix $C = [C_1, C_2, C_3, ... C_n]^T$
	- $C$  is $n * m$, and $S$ is $m * n_f$

	$$
	C = \left[
	\begin{array}{cccc}
	  c_{1,1} & c_{1,2} & \cdots & c_{1, m}\\
	  c_{2,1} & c_{2,2} & \cdots & c_{2, m}\\
	  \vdots & \vdots & \ddots & \vdots\\
	  c_{n,1} & c_{n,2} & \cdots & c_{n, m}
	\end{array}
	\right], 
	S = \left[
	\begin{array}{cccc|ccc}
	  f_1^{[1]} & f_2^{[1]} & \cdots & f_{n_f}^{[1]} & c_1^{[1]} & \cdots & c_{n_{sources}}^{[1]}\\
	  f_1^{[2]} & f_2^{[2]} & \cdots & f_{n_f}^{[2]} & c_1^{[2]} & \cdots & c_{n_{sources}}^{[2]}\\
	  \vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots\\
	  f_1^{[m]} & f_2^{[m]} & \cdots & f_{n_f}^{[m]}  & c_1^{[m]} & \cdots & c_{n_{sources}}^{[m]}\\
	\end{array}
	\right]\\
	
	\hat{S} = CS = \left[
	\begin{array}{cccc|ccc}
	  \hat{f}_1^{[1]} & \hat{f}_2^{[1]} & \cdots & \hat{f}_{n_f}^{[1]} & \hat{c}_1^{[1]} & \cdots & \hat{c}_{n_{sources}}^{[1]}\\
	  \hat{f}_1^{[2]} & \hat{f}_2^{[2]} & \cdots & \hat{f}_{n_f}^{[2]} & \hat{c}_1^{[2]} & \cdots & \hat{c}_{n_{sources}}^{[2]}\\
	  \vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots\\
	  \hat{f}_1^{[m]} & \hat{f}_2^{[m]} & \cdots & \hat{f}_{n_f}^{[m]}  & \hat{c}_1^{[m]} & \cdots & \hat{c}_{n_{sources}}^{[m]}\\
	\end{array}
	\right]
	$$
	
- Feature engineering using NCBI Taxonomy database and Tensorflow parallel architecture.

$$
F^{(g)} = \left[
\begin{array}{c}
  \vec{f}^{[1](g)}\\
  \vec{f}^{[2](g)}\\
  \vdots\\
  \vec{f}^{[m](g)}\\
\end{array}
\right]
= \left[
\begin{array}{cccc|ccc}
  f_1^{[1](g)} & f_2^{[1](g)} & \cdots & f_{n_f}^{[1](g)}\\
  f_1^{[2](g)} & f_2^{[2](g)} & \cdots & f_{n_f}^{[2](g)}\\
  \vdots & \vdots & \ddots & \vdots\\
  f_1^{[m](g)} & f_2^{[m](g)} & \cdots & f_{n_f}^{[m](g)}\\
\end{array}
\right], \\
W^{(g\rightarrow r)} = \left[
\begin{array}{cccc|ccc}
  w^{(g_1\rightarrow r_1)} & w^{(g_1\rightarrow r_2)} & \cdots & w^{(g_1\rightarrow r_{n_f})}\\
  w^{(g_2\rightarrow r_1)} & w^{(g_2\rightarrow r_2)} & \cdots & w^{(g_2\rightarrow r_{n_f})}\\
  \vdots & \vdots & \ddots & \vdots\\
  w^{(g_{n_f}\rightarrow r_1)} & w^{(g_{n_f}\rightarrow r_2)} & \cdots & w^{(g_{n_f}\rightarrow r_{n_f})}\\
\end{array}
\right]
$$

$$
F^{(r)} = F^{(g)}W^{(g\rightarrow r)} = \left[
\begin{array}{cccc|ccc}
  f_1^{[1](r)} & f_2^{[1](r)} & \cdots & f_{n_f}^{[1](r)}\\
  f_1^{[2](r)} & f_2^{[2](r)} & \cdots & f_{n_f}^{[2](r)}\\
  \vdots & \vdots & \ddots & \vdots\\
  f_1^{[m](r)} & f_2^{[m](r)} & \cdots & f_{n_f}^{[m](r)}\\
\end{array}
\right], r \in \{sk, p, c, o, f, g\}
$$



this is what dense layer of tf2 is doing $X_{next} = X_{prev}W+b$
$$
w^{(g_i\rightarrow r_j)} = w^{(g_j\rightarrow r_i)} = \left\{ 
\begin{aligned}
1, taxon^{(r)}(i) = taxon^{(r)}(j) \\
0, taxon^{(r)}(i) \neq taxon^{(r)}(j) \\
\end{aligned}
\right.
$$
NumPy 3D-matrix multiplication:  1.broadcasting over axis 0, apply matmul to each individual pair of matrix.

##### Post processing 

output layer: non-negative constraints for weights. $\Rightarrow y > 0$

- source tracking: 
	- applying $relu$ activation to Unknown source contribution. 
- Disease (exclude phenotype "health"):
	- applying $tanh$ activation.


## Conclusion

## Code Availability

## Acknowledgements

## References