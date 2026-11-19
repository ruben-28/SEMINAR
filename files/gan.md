
<style>
h2 { break-after: avoid; page-break-after: avoid; }
figure { break-inside: avoid; page-break-inside: avoid; }
</style>

# Report 1 – Software Engineering Seminar (157119)

**Paper studied:** Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y. (2014). *Generative Adversarial Nets*. Advances in Neural Information Processing Systems 27 (NIPS 2014). <https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf>

**Report authors:** Ethan Sarfati, Ruben Bensimon | **Submission date:** 19 November 2026

*Page numbers refer to the PDF of the paper (9 pages).*

## 1. Why did we choose this paper?

We chose *Generative Adversarial Nets* because we wanted to understand how a computer learns to create images, rather than simply use a ready-made AI model. The paper introduced an approach that became influential, with 125,493 Google Scholar citations when we checked on 11 November 2026. It also seemed realistic to reproduce: MNIST is public, the original networks are relatively small, the authors linked their code (p. 1), and Table 1 (p. 6) gives us a result to compare against. Training two networks that compete is not straightforward, so we also liked the technical challenge. Our backup choice was *GloVe* (Pennington et al., 2014), since its published word vectors allow quantitative tests without retraining a huge model; however, implementing a GAN seemed like a better learning experience.

## 2. What is the paper's main idea?

The paper introduces a way to generate data using two neural networks trained against each other. The generator, G, takes random noise and tries to create realistic samples, while the discriminator, D, decides whether an example is real or generated. The authors compare this to counterfeiters trying to fool the police (p. 1). They describe the process as a minimax game (Equation 1, p. 3). In the ideal case, the generated distribution matches the real one and D cannot do better than guessing with probability 0.5 (Theorem 1, p. 5). Both networks can be trained with backpropagation, without the Markov-chain sampling required by some earlier models. Experiments on MNIST, TFD and CIFAR-10 show promising, though not equally convincing, results (Table 1 and Figure 2, pp. 6–7).

## 3. What problem does the paper address and what contributions does it announce?

Earlier deep generative models often required probability calculations that were difficult to evaluate, including approximations based on Markov chains. The authors also note that techniques such as ReLU and maxout, successful in discriminative networks, were harder to exploit in generative models (Introduction, p. 1). GANs offer another way to train a generator: instead of calculating the data likelihood directly, they use a discriminator's feedback.

The paper does not provide a numbered contribution list, but we identified these contributions:

1. A generative training framework based on competition between G and D (Section 3, p. 3).
2. A theoretical analysis of the ideal optimum, where `p_g = p_data` and `D = 0.5` (Section 4, p. 5).
3. A practical algorithm based on alternating updates and backpropagation, without Markov chains (Algorithm 1, p. 4).
4. Experimental evidence that the method can compete with existing generative approaches (Section 5, p. 6).

We consider the first contribution the most important, since the adversarial setup is what makes the method different.

## 4. What makes this paper unique compared with earlier work?

Deep Boltzmann Machines require approximations to optimize a likelihood that is difficult to compute. Generative Stochastic Networks (GSNs) can use backpropagation but still rely on Markov chains for sampling. GANs avoid both requirements (Related Work, pp. 2–3). Variational Autoencoders also involve two networks, but their second network performs approximate inference rather than trying to distinguish real data from generated samples.

The authors also compare GANs with Noise-Contrastive Estimation, which distinguishes data from noise but requires evaluating density ratios, and with Predictability Minimization, where the competition between networks is only a regularizer. In GANs, that competition is the training objective itself. Table 2 (p. 8) summarizes these differences.

## 5. What is the methodology and what data is used?

The experiments use MNIST handwritten digits, the Toronto Face Database (TFD), and CIFAR-10 images (Section 5, p. 6). The paper names the datasets but gives little detail about their sizes or preprocessing. We downloaded MNIST and CIFAR-10 with torchvision and checked 60,000 and 50,000 training images respectively on 15 November 2026. We have not obtained TFD because access appears restricted.

The generator uses ReLU and sigmoid activations, with random noise supplied at the input. The discriminator uses maxout and dropout. Algorithm 1 alternates updates of D and G, with `k = 1` and momentum in the experiments. The authors recommend training G to maximize `log D(G(z))` because minimizing `log(1 − D(G(z)))` can lead to weak gradients early on (pp. 3–4).

For quantitative evaluation, they fit Gaussian Parzen windows to generated examples and estimate test-set log-likelihood, selecting σ by cross-validation. MNIST and TFD are evaluated numerically, but CIFAR-10 is assessed visually. Figures 2 and 3 (p. 7) show generated samples, nearest training examples and transitions between points in the noise space.

## 6. Do the results support the claims? What are the limitations?

Table 1 (p. 6) reports the following Parzen-window log-likelihood estimates; higher is better under this metric.

| Model | MNIST | TFD |
|---|---:|---:|
| DBN | 138 ± 2 | 1909 ± 66 |
| Stacked CAE | 121 ± 1.6 | 2110 ± 50 |
| Deep GSN | 214 ± 1.1 | 1890 ± 29 |
| **GAN** | **225 ± 2** | **2057 ± 26** |

The GAN obtains the best MNIST result, but not the best TFD result: Stacked CAE scores 2110 ± 50. We therefore agree with the authors' more limited claim that adversarial training is competitive, not universally superior. Figure 2 is convincing on MNIST: the digits are recognizable and the nearest training examples resemble them without being identical. The TFD faces are less sharp, and the CIFAR-10 images often look more like textures than identifiable objects.

<figure style="margin:0; break-inside: avoid; page-break-inside: avoid;">
<img src="https://dennybritz.com/img/deep-learning-most-important-ideas/early-gan-images.png" alt="Figure 2 of the paper" width="340">
<figcaption><em>Figure 2 (p. 7): a) MNIST, b) TFD, c) CIFAR-10 (fully connected model), d) CIFAR-10 (convolutional discriminator, "deconvolutional" generator).</em></figcaption>
</figure>

Figure 3 shows a gradual change between generated digits as the input vector changes. That suggests the generator learned a continuous representation rather than storing a fixed list of images.

<figure style="margin:0; break-inside: avoid; page-break-inside: avoid;">
<img src="https://tyami.github.io/assets/images/post/DL/2020-10-17-GAN-DCGAN-theory/2020-10-17-GAN-DCGAN-theory-5-GAN-interpolation.png" alt="Figure 3 of the paper" width="340">
<figcaption><em>Figure 3 (p. 7): digits obtained by linear interpolation between coordinates in z space.</em></figcaption>
</figure>

The authors acknowledge that GANs do not give an explicit density `p_g(x)`, that G and D need to remain balanced, and that the theoretical proof assumes an ideal non-parametric model (Sections 4 and 6). They also warn about Parzen estimates in high dimensions. We would add that the paper does not report variability across random seeds or a systematic hyperparameter study, and CIFAR-10 has no numerical score. Later work addressed some open questions through conditional GANs, DCGAN and WGAN (Section 7; Mirza & Osindero, 2014; Radford et al., 2015; Arjovsky et al., 2017).

## 7. Will we be able to reproduce the results?

We plan to reproduce the MNIST result from Table 1: **225 ± 2**. The authors' code is linked in footnote 1 (p. 1): https://github.com/goodfeli/adversarial. When we checked it on 15 November 2026, its latest commit was dated 6 June 2018, while the main implementation dated back to 2014. It relies on old libraries such as Theano and Pylearn2, so we will probably implement the model in PyTorch. The repository's `parzen_ll.py` script should help us reproduce the evaluation.

MNIST is available in IDX files and CIFAR-10 in Python or binary batches; we already downloaded both. We will focus on MNIST, since TFD is not currently available to us. The paper leaves out several details needed for exact reproduction: layer widths, maxout pieces, dropout values, learning rate, momentum value, iteration count, preprocessing, Parzen σ grid, sample count and random seed. We will compare the original code with the paper before choosing these parameters.

We also ran a preliminary MNIST test on a computer with an Intel Core i5-1135G7 and 16 GB RAM, without a GPU. Training a small perceptron for one epoch took about 25 seconds. GAN training updates two networks, so we initially estimate one to two minutes per epoch, although the real time may differ. We can use Google Colab if repeated experiments become too slow. The difficult part will probably be training stability rather than downloading the data.

## 8. What is the paper's influence over time?

The paper appeared at NIPS 2014 (now NeurIPS), a CORE A* conference. Google Scholar showed **125,493 citations** across its versions on 11 November 2026: 100,712 for the conference version and 25,967 for the *Communications of the ACM* version. The annual graph grew strongly from 2016 to 2020 and peaked in 2025. Since 2026 is not finished, and Scholar updates can be delayed, its lower bar does not establish a lasting decline.

<figure style="margin:0; break-inside: avoid; page-break-inside: avoid;">
<img src="figures/scholar.png" alt="Google Scholar screenshot" width="300">
<figcaption><em>Google Scholar screenshot (retrieved on 11 November 2026).</em></figcaption>
</figure>

We examined two later papers. Radford et al. (2015) developed DCGAN, using convolutional architectures to improve GAN training and image generation; Google Scholar listed 22,748 citations on 15 November 2026. Theis, van den Oord and Bethge (2016), with 1,590 citations, criticized Parzen-window evaluations and showed why they can be misleading in high dimensions. One work extends the original GAN, while the other questions a metric used in its experiments.

The paper remains relevant even though other image-generation methods have advanced. Dhariwal and Nichol (2021), for instance, reported diffusion-model results stronger than leading GANs of that period, including FID 2.97 on ImageNet 128 × 128. Still, the original adversarial training idea remains important for understanding later research.

## 9. Reflection: what do we think of the quality of the writing?

We found the paper fairly clear for a nine-page article. The counterfeiters-and-police example (p. 1) and Figure 1 (p. 4) helped us understand the training process, while Table 2 (p. 8) made the differences from earlier work easier to follow. Section 4 was much harder because it assumes knowledge of KL divergence, Jensen–Shannon divergence and convexity, and the proof of Proposition 2 is brief. We also noticed small inconsistencies: Algorithm 1 uses the saturating generator objective although the text recommends a different one, the noise distribution is written `p_g(z)` instead of `p_z`, and the proof refers to `p_x` where `p_data` would be expected. Some experimental settings are left to the code, which makes reproduction harder. Overall, the main idea is presented well, but the technical sections require careful reading.

## Use of a large language model (LLM)

We used an LLM to help understand difficult English and mathematical terms, find related papers, organize notes, prepare drafts and improve the English translation and formatting. We reviewed the technical claims and edited the text using the original paper before submitting the report.
