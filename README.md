# Report 1 – Software Engineering Seminar (157119)

**Paper:** Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y. (2014). *Generative Adversarial Nets*. Advances in Neural Information Processing Systems 27 (NIPS 2014).

https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf

**Authors:** Ethan Sarfati, Ruben Bensimon

**Submission date:** 19 November 2026

Page references below refer to the original 9-page paper.

## 1. Why did we choose this paper?

We chose *Generative Adversarial Nets* because we wanted to understand how generative AI actually works, rather than just use existing tools. GANs are one of the important developments in this field, and this paper is where the idea was first introduced. It has received 125,493 citations according to Google Scholar (11 November 2026), which also shows how much influence it has had.

Another reason was that the paper seemed possible to reproduce with the resources we have. It uses relatively small neural networks, the MNIST dataset is publicly available, and the authors published their code. There is also a numerical result in Table 1 (p. 6) that we can use to compare our implementation with theirs.

We were also interested in the training process itself. GANs are known for being difficult to train because the two networks have to improve together. We thought it would be interesting to see whether we could get this process to work and generate recognizable images ourselves.

Our second choice was *GloVe* (Pennington et al., 2014). Its results can be evaluated using word similarity and analogy tasks, and pre-trained word vectors are available. This would make it a safer option if we could not reproduce the GAN results.

However, we preferred GANs for two reasons. First, MNIST is small enough for us to train a model without needing expensive hardware, while reproducing the original GloVe training would require much larger text datasets. Second, with GANs we would need to implement and train the model ourselves. With GloVe, evaluating pre-trained vectors would not give us the same practical experience.

## 2. What is the paper's main idea?

The main idea is to train a model to generate new data by making two neural networks compete against each other.

The first network is the generator, G. It receives random noise as input and tries to produce data that looks real, such as images of handwritten digits. The second network is the discriminator, D. Its job is to distinguish between real images from the training dataset and fake images produced by G.

The authors compare this process to counterfeiters and the police (p. 1). The counterfeiters try to create convincing fakes, while the police learn how to identify them. As both improve, the fakes become harder to detect.

Mathematically, the training is described as a minimax game between G and D (Equation 1, p. 3). In the ideal situation, the generator learns the same probability distribution as the real data. At that point, the discriminator cannot distinguish real from generated samples and outputs a probability of 0.5 (Theorem 1, p. 5).

An important aspect of the method is that both networks can be trained using backpropagation. Unlike some earlier generative models, GANs do not require Markov chains during training or sample generation (Section 6, p. 6).

The authors test the method on MNIST, the Toronto Face Database (TFD), and CIFAR-10. Their experiments show that the model can generate recognizable images and obtain results comparable to other generative models available at the time.

## 3. What problem does the paper address, and what are its contributions?

Before GANs, training deep generative models was often complicated. The authors explain two main difficulties in the introduction (p. 1).

The first was that many generative models relied on probability calculations that were difficult or impossible to compute exactly. These calculations had to be approximated, sometimes using expensive methods such as Markov chain sampling.

The second difficulty was related to neural network architectures. Techniques such as ReLU and maxout had worked well in discriminative models, but using them effectively in generative models was less straightforward.

The paper proposes a different training approach that avoids these problems. Instead of explicitly calculating the probability of the data, it trains a generator using feedback from a discriminator.

We identified four main contributions:

1. The introduction of adversarial training, where two neural networks compete to improve a generative model (Section 3, p. 3).
2. A theoretical explanation showing that, under ideal conditions, the generator can learn the real data distribution and the discriminator will output 0.5 (Section 4, p. 5).
3. A training algorithm that uses backpropagation without requiring Markov chains or a separate inference procedure (Algorithm 1, p. 4).
4. Experimental results showing that this approach can compete with existing generative methods (Section 5, p. 6).

For us, the most important contribution is the adversarial training idea. The mathematical analysis helps explain why the method should work, but the idea of training one network against another is what makes the paper particularly interesting.

## 4. What makes this paper different from earlier work?

The authors discuss several previous approaches in Section 2 (pp. 2–3) and compare them in Table 2 (p. 8).

Deep Boltzmann Machines (Salakhutdinov & Hinton, 2009) require approximations to compute the gradients of their likelihood function. GANs avoid this because they do not explicitly calculate a likelihood.

Generative Stochastic Networks, or GSNs (Bengio et al., 2014), can be trained using backpropagation, but they still depend on Markov chains to generate samples. GANs do not need these chains.

Variational Autoencoders, or VAEs (Kingma & Welling, 2014; Rezende et al., 2014), also use two networks. However, their second network is used for approximate inference rather than discrimination. The paper also points out a difference concerning discrete variables: the proposed GAN approach cannot directly model discrete data, while the VAE formulations discussed have limitations with discrete latent variables.

Noise-Contrastive Estimation (Gutmann & Hyvärinen, 2010) learns by distinguishing real data from noise. This sounds similar to GANs, but NCE requires evaluating a ratio of probability densities, while GANs learn the discrimination through a neural network.

The authors also mention Predictability Minimization (Schmidhuber, 1992), which uses competing networks. However, its objective is different: the competition acts as a regularizer rather than a minimax game for learning a generative distribution.

Finally, adversarial examples (Szegedy et al.) are also discussed, but they are used to study and exploit errors in classification systems, not to train a generator.

What makes GANs different is that the competition between the networks becomes the actual training method. This allows the model to learn to generate data without explicitly calculating its probability distribution.

## 5. What methodology and data are used?

The authors evaluate their approach on three datasets: MNIST, the Toronto Face Database (TFD), and CIFAR-10 (Section 5, p. 6).

MNIST contains handwritten digits, TFD contains face images, and CIFAR-10 contains small images of objects and scenes. The paper identifies these datasets but does not provide much detail about their sizes or preprocessing.

We downloaded MNIST and CIFAR-10 using torchvision and checked that they could be loaded correctly on 15 November 2026. We obtained 60,000 MNIST training images and 50,000 CIFAR-10 training images. We have not obtained TFD because, according to the information we found, access to it is restricted.

### Network architecture

The generator uses a combination of ReLU and sigmoid activation functions. Random noise is provided only at its input layer.

The discriminator uses maxout activation functions and dropout during training (p. 6).

### Training

The training procedure is described in Algorithm 1 (p. 4). The discriminator is updated for k steps, followed by one update of the generator. In the experiments, the authors use k = 1 and momentum.

The paper also discusses a practical problem with the generator's loss function. Minimizing log(1 − D(G(z))) can give very small gradients when the discriminator is already good at recognizing fake samples. To avoid this issue, the authors recommend maximizing log D(G(z)) instead (p. 3). Both objectives have the same desired solution, although Algorithm 1 is written using the original objective.

### Evaluation

To evaluate the generated samples numerically, the authors use Gaussian Parzen windows. They fit a probability density estimate to samples produced by the generator and measure the mean log-likelihood of test images. The parameter σ is selected using cross-validation.

For MNIST, the evaluation uses real-valued images, and the standard error is calculated across test examples. For TFD, it is calculated across folds, with a separate σ for each fold (Table 1, p. 6).

CIFAR-10 is evaluated visually rather than with a numerical score.

The authors also show randomly generated images alongside their nearest examples from the training dataset (Figure 2, p. 7). This makes it possible to see whether the generated images are similar to real ones without simply being copies.

Figure 3 (p. 7) shows interpolations between points in the generator's input space. These examples help illustrate what the network has learned.

## 6. Do the results support the authors' claims? What are the limitations?

The main quantitative results appear in Table 1 (p. 6). The values are estimated log-likelihoods obtained using Parzen windows, so higher values indicate better results under this particular evaluation method.

| Model | MNIST | TFD |
|---|---|---|
| DBN | 138 ± 2 | 1909 ± 66 |
| Stacked CAE | 121 ± 1.6 | 2110 ± 50 |
| Deep GSN | 214 ± 1.1 | 1890 ± 29 |
| GAN | 225 ± 2 | 2057 ± 26 |

On MNIST, the GAN achieves the highest value, 225 ± 2, compared with 214 ± 1.1 for Deep GSN. The difference is larger than the reported standard errors, so the result is encouraging.

On TFD, the GAN performs better than DBN and Deep GSN, but Stacked CAE obtains a higher score of 2110 ± 50. The uncertainty ranges are relatively large, so the difference should be interpreted carefully.

We think the results support the main claim of the paper: adversarial training is a viable method for learning generative models. However, the experiments do not show that GANs are better than every previous method, and the authors do not make that claim.

The visual results are also important because the numerical metric does not tell us everything about the quality of the images.

### Analysis of Figure 2

Figure 2 (p. 7) presents generated images for MNIST, TFD and CIFAR-10.

For MNIST (a), most digits are clear and recognizable. We can identify digits such as 7, 3, 9, 0, 1, 2, 6 and 8. The writing styles also vary, suggesting that the generator has learned more than one fixed example. The closest training images look similar but are not identical.

For TFD (b), the faces are recognizable and show different expressions, although some details are blurry.

The CIFAR-10 results (c and d) are less convincing. Many of the images resemble colored textures rather than identifiable objects. The version with a convolutional discriminator and a deconvolutional generator (d) appears slightly more structured than the fully connected version (c), but the image quality remains limited.

Overall, MNIST gives the clearest evidence that the model can generate realistic samples. The results on faces are reasonable, while CIFAR-10 seems much more difficult for the architectures tested.

### Analysis of Figure 3

Figure 3 (p. 7) shows images created by interpolating between different points in the noise space z.

The generated digits change gradually, for example between shapes resembling 1 and 5, and between 7, 9 and 1. These transitions are interesting because the generator is not simply switching between unrelated images. It has learned a representation in which nearby input values can produce related outputs.

### Limitations

The authors acknowledge several limitations.

One is that the model does not provide an explicit probability density p_g(x). The generator and discriminator also need to remain balanced during training. If one becomes too strong compared with the other, learning may become difficult. The authors describe a possible failure as the "Helvetica scenario", where many different inputs produce the same output (Section 6, p. 6).

Another limitation is the evaluation method. Parzen-window estimates can have high variance and become unreliable for high-dimensional data. The authors recognize this problem, even though they consider it the best available evaluation method for their experiments (Section 5, p. 6).

The theoretical convergence results also assume an ideal non-parametric setting. They do not guarantee that the neural networks used in actual experiments will converge in the same way.

We noticed some additional limitations. The paper does not report results across multiple random seeds, making it difficult to know how stable the training process is. There is also no detailed study of how the hyperparameters affect performance.

Furthermore, MNIST and TFD use different ways of calculating standard errors, which complicates direct comparisons. CIFAR-10 has no quantitative result, so its evaluation depends entirely on the images shown.

### Possible improvements

In Section 7 (pp. 6–7), the authors suggest several possible directions, including conditional generation, approximate inference, semi-supervised learning and more efficient training.

Some of these directions were later explored in other papers. Examples include Conditional GANs (Mirza & Osindero, 2014), convolutional GANs such as DCGAN (Radford et al., 2015), and Wasserstein GANs (Arjovsky et al., 2017), which address training stability.

These later developments also show that the original method left room for improvement, particularly in the architecture and training procedure.

## 7. Will we be able to reproduce the results?

Our goal is to reproduce the MNIST result from Table 1, where the GAN obtains an estimated log-likelihood of 225 ± 2.

The authors provide a link to their implementation in footnote 1 (p. 1):

https://github.com/goodfeli/adversarial

We checked the repository on 15 November 2026. Its most recent commit was from 6 June 2018 and only merged a typo correction. The main code has not been substantially updated since June 2014.

The original implementation depends on Pylearn2 and Theano, which are now outdated. The repository documentation also mentions that exact reproduction may depend on the library versions and the original hardware, including GTX-580 GPUs.

Because of this, we think reimplementing the experiment in PyTorch is more practical than trying to run the original code.

The repository also includes a script called `parzen_ll.py`, which is useful because it shows how the authors calculated their evaluation metric.

### Data availability

MNIST and CIFAR-10 are publicly available. We successfully downloaded them through torchvision, so obtaining the data should not be a problem.

TFD is less accessible. Based on our research, it is a restricted dataset, and we have not obtained it. We therefore plan to focus on reproducing the numerical MNIST result. If we have enough time, we would also like to generate and compare CIFAR-10 samples.

### Information missing from the paper

One difficulty is that the paper does not provide all the details needed to recreate the experiments.

It does not specify the exact number and size of the hidden layers, the number of maxout pieces, dropout rates, learning rate, momentum value, or total number of training iterations.

Some details of the preprocessing and evaluation are also missing, including the range of σ values used for Parzen estimation, the number of generated samples, the random seed and the training/validation split.

The paper specifies k = 1 and the use of momentum, but we will need the original code or additional experimentation to determine the remaining settings.

For example, later work by Theis et al. (2016) discusses Parzen-window evaluations using 10,000 generated samples. We may use this as a reference when implementing the evaluation, although we will need to check how closely it matches the original experiment.

### Computing resources

The authors acknowledge the computing resources provided by Compute Canada (p. 8). However, since MNIST is relatively small and the original models use multilayer perceptrons, we believe a reduced reproduction should be possible on a normal computer.

We performed a preliminary test using a small perceptron trained on MNIST. One epoch over 60,000 images took approximately 25 seconds on a computer with an Intel Core i5-1135G7 processor and 16 GB of RAM, without a dedicated GPU.

GAN training requires updating two networks, so we expect it to take longer. As an initial estimate, we think an epoch could take around one to two minutes, although the actual time will depend on the implementation and network architecture.

This seems manageable for our experiment. If repeated training runs become too slow, we can use Google Colab to access a GPU.

Our main concern is not the size of the dataset but whether we can train the two networks successfully and reproduce a result close to the one reported in the paper.

## 8. What has been the paper's influence over time?

The paper was published at NIPS 2014, now called NeurIPS, in volume 27. The conference has a CORE ranking of A*.

According to Google Scholar, which we consulted on 11 November 2026, the paper has 125,493 citations when the different versions are combined.

The NIPS 2014 version accounts for 100,712 citations across 60 versions, while the *Communications of the ACM* version from 2020 has 25,967 citations.

The citation graph shows rapid growth between 2016 and 2020, followed by slower growth in later years. The highest annual count shown is in 2025. The number for 2026 is lower, but the year is not finished, and Google Scholar does not necessarily update all citations immediately. We therefore cannot conclude that interest in the paper is declining.

Since its publication in 2014, the paper has received roughly 10,000 citations per year on average. This is a very high number and confirms that GANs have been widely studied and used.

### Examples of later research

We looked at two papers that cite the original GAN work for different reasons.

Radford et al. (2015) introduced DCGAN, which replaces the fully connected networks with convolutional architectures. Their paper had 22,748 citations on 15 November 2026.

The purpose was to improve image generation and representation learning using convolutional networks. The original GAN paper already included an experiment with convolutional components (Figure 2d), but DCGAN developed this approach further.

Theis, van den Oord and Bethge (2016) examined how generative models are evaluated. Their work had 1,590 citations when we checked.

They explain that Parzen-window estimates can be misleading in high-dimensional spaces. One of their examples shows that even a simple k-means-based model can outperform the true data distribution according to this metric. This is especially relevant because the original GAN paper uses Parzen windows for its numerical comparison.

These examples illustrate two different types of influence: one paper builds on the GAN architecture, while the other questions the reliability of the evaluation method used in the original experiments.

GANs have also influenced research in image translation, video prediction, representation learning, text generation and other fields.

### Are GANs still relevant?

GANs remain important, although they are no longer the only major approach to image generation.

For example, Dhariwal and Nichol (NeurIPS 2021) showed that diffusion models could achieve better image quality than the leading GAN models of that time. They reported a Fréchet Inception Distance (FID) of 2.97 on ImageNet at 128 × 128 resolution. Their paper had 16,883 citations.

Diffusion models have since become a major approach to image generation, but this does not make the original GAN paper irrelevant.

We think the paper is still worth studying because it introduced an important training principle. Even if newer methods perform better on some tasks, the idea of learning through competition between networks remains useful.

## 9. What do we think about the quality of the writing?

We found the paper generally clear, especially considering that it introduces an entirely new training method in only nine pages.

The explanation using counterfeiters and the police (p. 1) is helpful because it makes the basic idea understandable before introducing the mathematics. Figure 1 (p. 4) also helps explain how the generator's distribution changes during training.

We appreciated Table 2 (p. 8), which compares GANs with earlier generative models. Without this comparison, it would have been more difficult for us to understand what the proposed method actually changes.

The hardest section for us was Section 4, which contains the theoretical proofs. It assumes that the reader already understands Kullback-Leibler divergence, Jensen-Shannon divergence and some concepts from convex optimization.

In particular, the proof of Proposition 2 is quite short, and we had to spend additional time understanding the mathematical steps. We do not think the explanation is necessarily incorrect, but it requires more background knowledge than the earlier sections.

We also found a few inconsistencies while reading.

In Algorithm 1 (p. 4), the generator minimizes log(1 − D(G(z))), while the authors recommend maximizing log D(G(z)) in the main text (p. 3). The reason for the alternative objective is explained, but a reader implementing the algorithm could still be confused.

There are also small notation errors. The algorithm samples noise from `p_g(z)` even though the noise distribution was previously defined as `p_z`. In the proof of Proposition 2, the text refers to convergence to `p_x` instead of `p_data`.

We noticed a few typographical errors, including "indistiguishable", "succesful" and "an a range of datasets". References 4 and 5 also appear to be identical.

The biggest practical weakness, in our opinion, is the lack of experimental details. The paper explains the theoretical method relatively well, but someone trying to reproduce the results must rely heavily on the original code.

Despite these issues, we found the paper interesting and understandable overall. The main idea is explained clearly, even if the mathematics and implementation details require additional work.

## Use of a large language model (LLM)

We used an LLM while working on this report, mainly because some parts of the paper were difficult for us to understand in English, especially the mathematical explanations and technical vocabulary.

We also used it to clarify unfamiliar concepts and help find other papers related to GANs, including papers that cite the original work.

For the report itself, we used it to organize our notes, help prepare initial drafts, improve the wording and translate the text into English. We then reviewed the answers, compared the technical information with the original paper and made changes where necessary.

The analysis of the paper, our observations about the figures and the information about our reproduction plans were reviewed by us before submission.
