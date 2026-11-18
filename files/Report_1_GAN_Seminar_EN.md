<style>
h2 { break-after: avoid; page-break-after: avoid; }
figure { break-inside: avoid; page-break-inside: avoid; }
</style>

# Report 1 – Software Engineering Seminar (157119)

**Paper studied:** Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y. (2014). *Generative Adversarial Nets*. Advances in Neural Information Processing Systems 27 (NIPS 2014). <https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf>

**Report authors:** Ethan Sarfati, Ruben Bensimon | **Submission date:** 19 November 2026

*Page numbers refer to the PDF of the paper (9 pages).*

## 1. Why did we choose this paper?

We chose *Generative Adversarial Nets* because it is a foundational paper: it introduces a new framework for estimating generative models that opened up an entire field of research, as shown by its 125,493 citations on Google Scholar (retrieved on 11 November 2026). It is also well suited to a reproduction: the networks used are small multilayer perceptrons, MNIST is public, the authors' code is referenced (footnote 1, p. 1), and the paper provides a precise numerical result to reproduce (Table 1, p. 6). Our personal motivation is curiosity about generative AI: understanding how a model can learn to create new images, in a field at the heart of current AI. To this we add the appeal of a technical challenge: GANs have a reputation for being hard to train, and making them converge to a decent result is a real implementation challenge.

Our backup paper would be GloVe (Pennington et al., 2014). We choose GloVe as a backup because its evaluations (word analogies and word similarity) are quantitative and easy to reproduce with published vectors, which would limit the risk if the reproduction of the GAN failed. We nevertheless prefer the GAN paper, for two reasons. First, a small-scale reproduction is more natural there: the result of Table 1 is measured on MNIST, a small dataset, with small networks, whereas the published GloVe results rely on very large text corpora that are costly to retrain on an ordinary computer. Second, the GAN requires real implementation work: one has to write the adversarial training loop oneself, whereas with GloVe one can evaluate already pre-trained vectors without redoing the training.

## 2. What is the paper's main idea?

Goodfellow et al. propose an original way of teaching a computer to generate new data, for example images of handwritten digits. Two neural networks are made to work against each other: the generator starts from random noise and produces fake data, while the discriminator must guess whether a sample is real (from the training set) or fake. The authors compare this to counterfeiters facing a police force that tries to detect them (p. 1); as training progresses, each side improves and the generator produces increasingly realistic data. Formally, the two networks play a two-player minimax game (Equation 1, p. 3), whose ideal solution is unique: the generator reproduces the distribution of the real data and the discriminator can do no better than a random guess, i.e. a probability of ½ (Theorem 1, p. 5). With multilayer perceptrons, everything is trained by backpropagation: "Markov chains are never needed, only backprop is used to obtain gradients" (Section 6, p. 6). The experiments on MNIST, TFD and CIFAR-10 give visually convincing samples and a model that is competitive with the approaches of the time (Table 1, p. 6; Figure 2, p. 7).

## 3. What problem does the paper address and what contributions does it announce?

**Problem.** Before this paper, deep generative models had had less impact than discriminative models, for two reasons given by the authors: the difficulty of approximating many intractable probabilistic computations arising in maximum likelihood estimation, and the difficulty of exploiting, in a generative setting, the piecewise linear units (ReLU, maxout) that had made discriminative models successful (introduction, p. 1). The problem is therefore to train a deep generative model without going through these computations, which were approximated with costly techniques such as Markov chains.

**Contributions.** The paper does not contain an explicit list; we derive them from the abstract (p. 1) and from Sections 3 to 5:

1. A new framework for estimating generative models through an adversarial process between two networks (abstract; Section 3, p. 3).
2. A theoretical analysis: the solution is unique (`p_g = p_data` and D = ½) and the algorithm converges in the ideal case (Section 4, p. 5).
3. A training algorithm based on backpropagation alone, with no Markov chains or inference during learning (Algorithm 1, p. 4; Section 6, p. 6).
4. Experiments with competitive results, without claiming to do better than existing methods (Section 5, p. 6).

In our view, the main contribution is the first one (the adversarial framework), because it opened the way to a whole family of models.

## 4. What makes this paper unique compared with earlier work?

To position their work, the authors compare it with several existing approaches (Section 2, p. 2-3; summarized in Table 2, p. 8). Deep Boltzmann machines (Salakhutdinov & Hinton, 2009) are trained by maximizing an intractable likelihood whose gradient must be approximated in many ways; GANs have no explicit likelihood to compute. Generative stochastic networks, or GSNs (Bengio et al., 2014), are trained by exact backpropagation but use Markov chains to generate samples; GANs eliminate these chains.

Variational autoencoders, or VAEs (Kingma & Welling, 2014; Rezende et al., 2014), also pair a generator with a second network, but that network performs approximate inference, which GANs do not need; in return, GANs cannot model discrete data, whereas VAEs cannot have discrete latent variables. Noise-contrastive estimation, or NCE (Gutmann & Hyvärinen, 2010), learns a model by distinguishing it from noise, but its "discriminator" relies on a ratio of densities that has to be evaluated. Finally, predictability minimization (Schmidhuber, 1992) also pits two networks against each other, but only as a regularizer, with a scalar output and an optimization problem rather than a minimax game. The authors also distinguish their approach from adversarial examples (Szegedy et al.), which are not used to train a generative model (p. 3). The originality of GANs is therefore to replace the intractable likelihood and the Markov chains with a game between two networks trained by backpropagation.

## 5. What is the methodology and what data is used?

**Data** (Section 5, p. 6): MNIST, the Toronto Face Database (TFD) and CIFAR-10. The paper cites their sources but gives neither their size nor the preprocessing applied. We were able to download MNIST and CIFAR-10 (check done on 15 November 2026: 60,000 MNIST images and 50,000 CIFAR-10 training images loaded). According to our research, TFD is a private dataset with restricted access, which we have not obtained so far.

**Architecture** (p. 6): the generator uses a mixture of rectified linear (ReLU) and sigmoid activations, with noise injected only in the input layer; the discriminator uses *maxout* activations, with *dropout* during training.

**Training**: Algorithm 1 (p. 4), alternating k steps on D and one step on G, with k = 1 and momentum. The text (p. 3) recommends training G to maximize log D(G(z)) rather than to minimize log(1 − D(G(z))), because the latter saturates early in training while both have the same fixed point; Algorithm 1 itself remains written with the original form.

**Evaluation** (p. 6): quantitatively, a Gaussian Parzen window is fitted to samples generated by G, with σ chosen by cross-validation, and the mean log-likelihood on the test set is reported. For MNIST, the comparison is made on the real-valued version of the dataset and the standard error is computed across examples; for TFD, it is computed across folds with one σ per fold (Table 1 caption). CIFAR-10 is only evaluated qualitatively. Visually, Figure 2 shows randomly drawn samples together with the nearest training example, and Figure 3 shows a linear interpolation in z space (p. 7).

## 6. Do the results support the claims? What are the limitations?

**Results (Table 1, p. 6).** Log-likelihood estimated with Parzen windows (higher is better):

| Model | MNIST | TFD |
| --- | --- | --- |
| DBN | 138 ± 2 | 1909 ± 66 |
| Stacked CAE | 121 ± 1.6 | 2110 ± 50 |
| Deep GSN | 214 ± 1.1 | 1890 ± 29 |
| **GAN** | **225 ± 2** | 2057 ± 26 |

Overall, these results show that adversarial training works and reaches a decent result. It is the alternation of updates of D and G (Algorithm 1, p. 4) that gradually brings the generator closer to the data distribution, as illustrated by Figure 1 (p. 4): `p_g` approaches `p_data` and D tends towards ½. On MNIST, the GAN obtains the highest value (225, versus 214 for Deep GSN, 138 for DBN and 121 for Stacked CAE), and the gap with Deep GSN exceeds the standard errors. On TFD, the GAN (2057 ± 26) beats DBN (1909) and Deep GSN (1890) but not the Stacked CAE (2110 ± 50), whose gap is of the same order as the standard errors: the two are therefore comparable. The model is thus competitive with established approaches, which is what the authors claim without asserting superiority (Section 5, p. 6).

The visual results point in the same direction (Figures 2 and 3, p. 7). The samples in Figure 2 are randomly drawn and not cherry-picked, and the last column (yellow frame) shows the nearest training example, to check that the generator does not simply copy the training set.

<figure style="margin:0; break-inside: avoid; page-break-inside: avoid;">
<img src="figures/figure2.png" alt="Figure 2 of the paper" width="340">
<figcaption><em>Figure 2 (p. 7): a) MNIST, b) TFD, c) CIFAR-10 (fully connected model), d) CIFAR-10 (convolutional discriminator, "deconvolutional" generator).</em></figcaption>
</figure>

In Figure 2 a), the MNIST digits are sharp and recognizable (7, 3, 9, 0, 1, 2, 6, 8), with varied writing styles, and their nearest training neighbor resembles them without being identical: it is not a copy. For TFD (b), one recognizes faces with different expressions, despite a somewhat blurry appearance. For CIFAR-10 (c and d), the quality is clearly lower: the images mostly look like textures and colors of natural scenes, with no identifiable objects; the convolutional model (d) is slightly more structured than the fully connected model (c). In Figure 3, one moves gradually from a 1 to a 5, then from a 7 to a 9 and to a 1, with plausible intermediate digits, which suggests that the generator has learned a continuous latent space. The result is therefore convincing on MNIST, acceptable on TFD and more modest on CIFAR-10.

<figure style="margin:0; break-inside: avoid; page-break-inside: avoid;">
<img src="figures/figure3.png" alt="Figure 3 of the paper" width="340">
<figcaption><em>Figure 3 (p. 7): digits obtained by linear interpolation between coordinates in z space.</em></figcaption>
</figure>

**Limitations acknowledged by the authors:**

- There is no explicit representation of `p_g(x)`, and D and G must be well synchronized to avoid the "Helvetica scenario", in which G collapses many values of z onto the same x (Section 6, p. 6).
- The Parzen estimate has fairly high variance and performs poorly in high dimensions, while being the best method known to the authors (Section 5, p. 6).
- The proofs do not strictly apply to real networks, because they are carried out in the non-parametric setting (p. 5).

**Limitations we identify:** the metric is fragile, and the error bars for MNIST (across examples) and TFD (across folds) are not of the same nature; no variability due to training (random seeds) and no sensitivity study with respect to hyperparameters is reported; there is no quantitative result on CIFAR-10.

**Improvements (Section 7, p. 6-7).** The future work listed includes conditional models, learned approximate inference, modeling of conditionals, semi-supervised learning and improved training efficiency. Several have been carried out since: conditional GANs (Mirza & Osindero, 2014), convolutional architectures (Radford et al., 2015, DCGAN), improved stability (Arjovsky et al., 2017, WGAN).

## 7. Will we be able to reproduce the results?

**Reproduction target:** Table 1, MNIST column, "Adversarial nets" row: 225 ± 2.

**Code.** The address is given in footnote 1 (p. 1): github.com/goodfeli/adversarial. Consulted on 15 November 2026, the repository's last commit dates from 6 June 2018 (merge of a typo fix); the latest changes to the code itself date from June 2014. The repository is therefore no longer updated. According to the repository README (as reproduced in copies), the code is provided without support, exact reproduction depends on dependency versions and on hardware (GTX-580 cards), it requires Pylearn2 and Theano (old libraries), for a version from around June 2014, and the script `parzen_ll.py` performs the Parzen estimation. The experiment will therefore probably have to be reimplemented in PyTorch.

**Data.** MNIST (four files in IDX format: training and test images and labels) and CIFAR-10 (batches in Python "pickle" or binary format) are public. We downloaded them with torchvision. According to our research, TFD is a private dataset (restricted access, Susskind et al., 2010): we do not have it so far, and we will therefore reproduce the quantitative result on MNIST and, if time allows, the qualitative CIFAR-10 samples.

**Missing details in the paper.** Size and number of layers, number of maxout pieces, dropout rates, learning rate and momentum value, number of iterations, image preprocessing, σ grid, number of samples for Parzen, random seed, training/validation split. The paper only specifies k = 1 and the use of momentum; the rest is left to the code. In practice, Parzen windows in this type of work are fitted on 10,000 samples (Theis et al., 2016).

**Resources.** The authors thank Compute Canada for computing resources (p. 8). Since the networks are small perceptrons and MNIST is small, we think an ordinary computer will suffice. A training test of a small perceptron on MNIST (one epoch, i.e. 60,000 images) takes about 25 seconds on our computer, without a GPU (Intel Core i5-1135G7 processor, 16 GB of RAM). Since a GAN trains two networks, we estimate a cost of the order of two to four times higher, i.e. one to two minutes per epoch: this remains feasible on our machine, and Google Colab could be used to speed up repeated runs.

## 8. What is the paper's influence over time?

**Venue.** NIPS 2014 conference (now NeurIPS), volume 27. CORE ranking: A\*.

**Citations (Google Scholar, retrieved on 11 November 2026).**

- Total, merged versions: **125,493** citations; NeurIPS 2014 version: 100,712 (60 versions); Communications of the ACM 2020 version: 25,967.
- Trend: strong growth from 2016 to 2020, slower growth afterwards with a peak in 2025; the 2026 bar is lower, but the year is not over and Scholar updates with a delay, so one cannot speak of a decline. (screenshot below)
- Consistency: about 10,000 citations per year on average since June 2014, which is consistent with a foundational paper in a very broad and fast-growing field.

<figure style="margin:0; break-inside: avoid; page-break-inside: avoid;">
<img src="figures/scholar.png" alt="Google Scholar screenshot" width="300">
<figcaption><em>Google Scholar screenshot (retrieved on 11 November 2026).</em></figcaption>
</figure>

**Who cites it and why** (citation counts: Google Scholar, 15 November 2026):

- *Extends:* Radford et al. (arXiv 2015, DCGAN), cited 22,748 times, replace the perceptrons with convolutional networks, a direction already started by the convolutional model of Figure 2d of the paper; field: computer vision and representation learning.
- *Critiques:* Theis, van den Oord & Bethge (arXiv 2015, ICLR 2016), cited 1,590 times, show that Parzen estimates are unreliable in high dimension and that a simple k-means model can beat the true data distribution under this criterion; they recommend avoiding this metric, the very one used in Table 1. Field: evaluation of generative models.
- Fields of the citing papers in general: image generation and translation, sequence and text generation, online advertising, video prediction, etc.

**Relevance today.** Dhariwal & Nichol (NeurIPS 2021), cited 16,883 times, show that diffusion models reach better image quality than the best GANs of the time (FID of 2.97 on ImageNet 128×128). For image generation, diffusion models have therefore overtaken GANs, but the idea of the adversarial game is still used in many fields and the paper remains a major historical and conceptual reference.

## 9. Reflection: what do we think of the quality of the writing?

In our view, the paper is overall clear and well structured for its length (9 pages). The counterfeiters-and-police analogy (p. 1) and Figure 1 (p. 4) make the idea accessible, and Table 2 (p. 8) offers a good comparative summary. The hardest part is Section 4 (proofs): it assumes familiarity with the Kullback-Leibler divergence, the Jensen-Shannon divergence and notions of convexity, and the proof of Proposition 2 is very compact. The difficulty therefore comes mainly from the required background knowledge rather than from the style.

A few writing flaws harm precision:

- Algorithm 1 (p. 4) minimizes log(1 − D(G(z))), whereas the text (p. 3) recommends maximizing log D(G(z)).
- The noise there is drawn from "`p_g(z)`" whereas the notation defined is `p_z`.
- The proof of Proposition 2 concludes convergence to "`p_x`" instead of `p_data`.
- A few typos ("indistiguishable", "succesful", "an a range of datasets") and two identical references (references 4 and 5).
- The details of the experiments are deferred to the code rather than described in the text.

## Use of a large language model (LLM)

We used a large language model (LLM) in three ways. First, to help us translate the parts of the paper that we did not understand, because of the language barrier and the English mathematical terms. Second, to help with our research, for example to find the texts that cite our paper and to understand notions we did not know. Finally, it was useful for the layout and for obtaining clean writing. It also helped us structure the report and write first drafts of the answers to the questions, which we then reread, checked against the paper and adapted. It also helped us translate this report into English.
