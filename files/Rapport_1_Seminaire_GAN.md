<style>
h2 { break-after: avoid; page-break-after: avoid; }
figure { break-inside: avoid; page-break-inside: avoid; }
</style>

# Rapport 1 – Séminaire en génie logiciel (157119)

**Article étudié :** Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y. (2014). *Generative Adversarial Nets*. Advances in Neural Information Processing Systems 27 (NIPS 2014). <https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf>

**Auteurs du rapport :** ethan sarfati ruben bensimon | **Rendu :** 19.11.2026

*Les numéros de page renvoient au PDF de l'article (9 pages).*

## 1. Pourquoi avons-nous choisi cet article ?

Nous avons choisi *Generative Adversarial Nets* parce qu'il s'agit d'un article fondateur : il introduit un nouveau cadre d'estimation de modèles génératifs qui a ouvert tout un domaine de recherche, comme le montrent ses 125 493 citations sur Google Scholar (relevé du 11/11/2026). Il est aussi adapté à une reproduction : les réseaux utilisés sont de petits perceptrons multicouches, MNIST est public, le code des auteurs est référencé (note 1, p. 1) et l'article fournit un résultat chiffré précis à retrouver (Table 1, p. 6). Notre motivation personnelle est la curiosité pour l'IA générative : comprendre comment un modèle peut apprendre à créer de nouvelles images, dans un domaine au cœur de l'IA actuelle. À cela s'ajoute l'attrait d'un défi technique : les GAN ont la réputation d'être difficiles à entraîner, et les faire converger vers un résultat convenable est un vrai défi d'implémentation.

Notre article de secours serait GloVe (Pennington et al., 2014). Nous retenons GloVe comme secours parce que ses évaluations (analogies et similarité de mots) sont chiffrées et simples à reproduire avec des vecteurs publiés, ce qui limiterait le risque si la reproduction de GAN échouait. Nous préférons toutefois GAN, pour deux raisons. D'abord, la reproduction à petite échelle y est plus naturelle : le résultat de la Table 1 se mesure sur MNIST, un jeu de données de petite taille, avec de petits réseaux, alors que les résultats publiés de GloVe reposent sur de très grands corpus de texte, lourds à réentraîner sur un ordinateur ordinaire. Ensuite, GAN demande un vrai travail d'implémentation : il faut écrire soi-même la boucle d'entraînement adverse, alors qu'avec GloVe on peut évaluer des vecteurs déjà pré-entraînés sans refaire l'apprentissage.

## 2. Quelle est l'idée générale de l'article ?

Goodfellow et al. proposent une façon originale d'apprendre à un ordinateur à générer de nouvelles données, par exemple des images de chiffres manuscrits. On fait travailler deux réseaux de neurones l'un contre l'autre : le générateur part d'un bruit aléatoire et fabrique de fausses données, tandis que le discriminateur doit deviner si une donnée est vraie (issue du jeu d'entraînement) ou fausse. Les auteurs comparent cela à des faussaires face à une police qui cherche à les détecter (p. 1) ; au fil de l'entraînement, chacun progresse et le générateur produit des données de plus en plus réalistes. Formellement, les deux réseaux jouent à un jeu minimax à deux joueurs (équation 1, p. 3), dont la solution idéale est unique : le générateur reproduit la distribution des vraies données et le discriminateur ne peut plus faire mieux qu'un tirage au hasard, soit une probabilité de ½ (Théorème 1, p. 5). Avec des perceptrons multicouches, tout s'entraîne par rétropropagation : « Markov chains are never needed, only backprop is used to obtain gradients » (section 6, p. 6). Les expériences sur MNIST, TFD et CIFAR-10 donnent des échantillons visuellement convaincants et un modèle compétitif avec les approches de l'époque (Table 1, p. 6 ; Figure 2, p. 7).

## 3. Quel est le problème traité et quelles sont les contributions annoncées ?

**Problème.** Avant cet article, les modèles génératifs profonds avaient eu moins d'impact que les modèles discriminatifs, pour deux raisons indiquées par les auteurs : la difficulté d'approcher de nombreux calculs probabilistes intractables liés au maximum de vraisemblance, et la difficulté d'exploiter dans un cadre génératif les unités linéaires par morceaux (ReLU, maxout) qui avaient fait le succès des modèles discriminatifs (introduction, p. 1). Le problème est donc d'entraîner un modèle génératif profond sans passer par ces calculs, que l'on approchait par des techniques coûteuses comme les chaînes de Markov.

**Contributions.** L'article ne contient pas de liste explicite ; nous les dégageons du résumé (p. 1) et des sections 3 à 5 :

1. Un nouveau cadre d'estimation de modèles génératifs par un processus adverse entre deux réseaux (résumé ; section 3, p. 3).
2. Une analyse théorique : la solution est unique (`p_g = p_data` et D = ½) et l'algorithme converge dans le cas idéal (section 4, p. 5).
3. Un algorithme d'entraînement par rétropropagation seule, sans chaînes de Markov ni inférence pendant l'apprentissage (Algorithme 1, p. 4 ; section 6, p. 6).
4. Des expériences aux résultats compétitifs, sans prétention à faire mieux que l'existant (section 5, p. 6).

Selon nous, la contribution principale est la première (le cadre adverse), car elle a ouvert la voie à toute une famille de modèles.

## 4. Qu'est-ce qui rend cet article unique par rapport aux travaux antérieurs ?

Pour situer leur travail, les auteurs le comparent à plusieurs approches (section 2, p. 2-3 ; résumé dans la Table 2, p. 8). Les machines de Boltzmann profondes (Salakhutdinov & Hinton, 2009) s'entraînent par maximisation d'une vraisemblance intractable, dont le gradient doit être approché de multiples façons ; les GAN n'ont aucune vraisemblance explicite à calculer. Les réseaux stochastiques génératifs, ou GSN (Bengio et al., 2014), s'entraînent par rétropropagation exacte mais utilisent des chaînes de Markov pour générer des échantillons ; les GAN les suppriment.

Les auto-encodeurs variationnels, ou VAE (Kingma & Welling, 2014 ; Rezende et al., 2014), associent aussi un générateur à un second réseau, mais celui-ci réalise une inférence approchée, ce dont les GAN n'ont pas besoin ; en contrepartie, les GAN ne peuvent pas modéliser des données discrètes, alors que les VAE n'acceptent pas de variables latentes discrètes. L'estimation par contraste de bruit, ou NCE (Gutmann & Hyvärinen, 2010), apprend un modèle en le distinguant d'un bruit, mais son « discriminateur » repose sur un rapport de densités qu'il faut pouvoir évaluer. Enfin, la minimisation de prédictibilité (Schmidhuber, 1992) met aussi deux réseaux en compétition, mais seulement comme régularisateur, avec une sortie scalaire et un problème d'optimisation plutôt qu'un jeu minimax. Les auteurs distinguent aussi leur approche des exemples adverses (Szegedy et al.), qui ne servent pas à entraîner un modèle génératif (p. 3). L'originalité des GAN est donc de remplacer la vraisemblance intractable et les chaînes de Markov par un jeu entre deux réseaux entraîné par rétropropagation.

## 5. Quelle est la méthodologie et sur quelles données ?

**Données** (section 5, p. 6) : MNIST, Toronto Face Database (TFD) et CIFAR-10. L'article cite leurs sources mais ne donne ni leur taille ni le prétraitement appliqué. Nous avons pu télécharger MNIST et CIFAR-10 (vérification faite le 15.11.2026 : 60 000 images MNIST et 50 000 images d'entraînement CIFAR-10 chargées). D'après nos recherches, TFD est un jeu privé, à accès restreint, que nous n'avons pas obtenu à ce jour.

**Architecture** (p. 6) : le générateur utilise un mélange d'activations linéaires rectifiées (ReLU) et sigmoïdes, avec du bruit injecté seulement dans la couche d'entrée ; le discriminateur utilise des activations *maxout*, avec *dropout* à l'entraînement.

**Entraînement** : Algorithme 1 (p. 4), alternance de k étapes sur D et d'une étape sur G, avec k = 1 et du momentum. Le texte (p. 3) recommande d'entraîner G à maximiser log D(G(z)) plutôt qu'à minimiser log(1 − D(G(z))), car cette seconde forme sature au début de l'apprentissage alors que les deux ont le même point fixe ; l'Algorithme 1 lui-même reste écrit avec la forme d'origine.

**Évaluation** (p. 6) : quantitativement, une fenêtre de Parzen gaussienne est ajustée sur des échantillons générés par G, avec σ choisi par validation croisée, et l'on rapporte la log-vraisemblance moyenne sur le test. Pour MNIST, la comparaison porte sur la version à valeurs réelles du jeu et l'erreur standard est calculée sur les exemples ; pour TFD, elle est calculée sur les plis avec un σ par pli (légende de la Table 1). CIFAR-10 n'est évalué que qualitativement. Visuellement, la Figure 2 montre des échantillons tirés au hasard avec l'exemple d'entraînement le plus proche, et la Figure 3 une interpolation linéaire dans l'espace de z (p. 7).

## 6. Les résultats soutiennent-ils les affirmations ? Quelles limites ?

**Résultats (Table 1, p. 6).** Log-vraisemblance estimée par fenêtres de Parzen (plus c'est élevé, mieux c'est) :

| Modèle | MNIST | TFD |
| --- | --- | --- |
| DBN | 138 ± 2 | 1909 ± 66 |
| Stacked CAE | 121 ± 1,6 | 2110 ± 50 |
| Deep GSN | 214 ± 1,1 | 1890 ± 29 |
| **GAN** | **225 ± 2** | 2057 ± 26 |

Dans l'ensemble, ces résultats montrent que l'entraînement adverse fonctionne et permet d'arriver à un résultat convenable. C'est l'alternance des mises à jour de D et de G (Algorithme 1, p. 4) qui rapproche peu à peu le générateur de la distribution des données, ce qu'illustre la Figure 1 (p. 4) : `p_g` se rapproche de `p_data` et D tend vers ½. Sur MNIST, le GAN obtient la valeur la plus élevée (225, contre 214 pour Deep GSN, 138 pour DBN et 121 pour Stacked CAE), et l'écart avec Deep GSN dépasse les erreurs standard. Sur TFD, le GAN (2057 ± 26) dépasse DBN (1909) et Deep GSN (1890) mais pas le Stacked CAE (2110 ± 50), dont l'écart est du même ordre que les erreurs standard : les deux sont comparables. Le modèle est donc compétitif avec des approches établies, ce que les auteurs affirment sans prétendre à la supériorité (section 5, p. 6).

Les résultats visuels vont dans le même sens (Figures 2 et 3, p. 7). Les échantillons de la Figure 2 sont tirés au hasard et non sélectionnés, et la dernière colonne (cadre jaune) donne l'exemple d'entraînement le plus proche, pour vérifier que le générateur ne recopie pas le jeu d'entraînement.

<figure style="margin:0; break-inside: avoid; page-break-inside: avoid;">
<img src="figures/figure2.png" alt="Figure 2 de l'article" width="340">
<figcaption><em>Figure 2 (p. 7) : a) MNIST, b) TFD, c) CIFAR-10 (modèle entièrement connecté), d) CIFAR-10 (discriminateur convolutif, générateur « déconvolutif »).</em></figcaption>
</figure>

Sur la Figure 2 a), les chiffres MNIST sont nets et reconnaissables (7, 3, 9, 0, 1, 2, 6, 8), avec des styles variés, et leur voisin d'entraînement leur ressemble sans être identique : il ne s'agit pas d'une copie. Pour TFD (b), on reconnaît des visages d'expressions différentes, malgré un aspect un peu flou. Pour CIFAR-10 (c et d), la qualité est nettement plus faible : les images ressemblent surtout à des textures et des couleurs de scènes naturelles, sans objets identifiables ; le modèle convolutif (d) est un peu plus structuré que le modèle entièrement connecté (c). Sur la Figure 3, on passe progressivement d'un 1 à un 5, puis d'un 7 à un 9 et à un 1, avec des chiffres intermédiaires plausibles, ce qui suggère que le générateur a appris un espace latent continu. Le résultat est donc convaincant sur MNIST, correct sur TFD et plus modeste sur CIFAR-10.

<figure style="margin:0; break-inside: avoid; page-break-inside: avoid;">
<img src="figures/figure3.png" alt="Figure 3 de l'article" width="340">
<figcaption><em>Figure 3 (p. 7) : chiffres obtenus par interpolation linéaire entre des coordonnées de l'espace z.</em></figcaption>
</figure>

**Limites reconnues par les auteurs :**

- Pas de représentation explicite de `p_g(x)`, et D et G doivent être bien synchronisés pour éviter le « scénario Helvetica », où G fait s'effondrer beaucoup de valeurs de z vers le même x (section 6, p. 6).
- L'estimation de Parzen a une variance assez élevée et fonctionne mal en grande dimension, tout en étant la meilleure méthode connue des auteurs (section 5, p. 6).
- Les preuves ne s'appliquent pas strictement aux réseaux réels, car elles sont faites dans le cadre non paramétrique (p. 5).

**Limites que nous identifions :** la métrique est fragile, et les barres d'erreur de MNIST (sur les exemples) et de TFD (sur les plis) ne sont pas de même nature ; aucune variabilité due à l'entraînement (graines aléatoires) ni étude de sensibilité aux hyperparamètres n'est rapportée ; il n'y a pas de résultat quantitatif sur CIFAR-10.

**Améliorations (section 7, p. 6-7).** Les travaux futurs listés sont les modèles conditionnels, l'inférence approchée apprise, la modélisation des conditionnelles, l'apprentissage semi-supervisé et l'amélioration de l'efficacité de l'entraînement. Plusieurs ont été réalisés depuis : GAN conditionnels (Mirza & Osindero, 2014), architectures convolutives (Radford et al., 2015, DCGAN), amélioration de la stabilité (Arjovsky et al., 2017, WGAN).

## 7. Pourrons-nous reproduire les résultats ?

**Cible de reproduction :** Table 1, colonne MNIST, ligne « Adversarial nets » : 225 ± 2.

**Code.** L'adresse est donnée en note 1 (p. 1) : github.com/goodfeli/adversarial. Consulté le 15.11.2026, le dernier commit du dépôt date du 6 juin 2018 (fusion d'une correction de coquille) ; les dernières modifications du code lui-même datent de juin 2014. Le dépôt n'est donc plus mis à jour. D'après le README du dépôt (tel que repris par des copies), le code est fourni sans support, la reproduction exacte dépend des versions des dépendances et du matériel (cartes GTX-580), il requiert Pylearn2 et Theano (anciennes bibliothèques), pour une version d'environ juin 2014, et le script `parzen_ll.py` réalise l'estimation par Parzen. Il faudra donc probablement réimplémenter l'expérience en PyTorch.

**Données.** MNIST (quatre fichiers au format IDX : images et étiquettes d'entraînement et de test) et CIFAR-10 (lots au format Python « pickle » ou binaire) sont publics. Nous les avons téléchargés avec torchvision. D'après nos recherches, TFD est un jeu privé (accès restreint, Susskind et al., 2010) : nous ne l'avons pas à ce jour et nous reproduirons donc le résultat quantitatif sur MNIST et, si le temps le permet, les échantillons qualitatifs de CIFAR-10.

**Détails manquants dans l'article.** Taille et nombre des couches, nombre de pièces maxout, taux de dropout, taux d'apprentissage et valeur du momentum, nombre d'itérations, prétraitement des images, grille de σ, nombre d'échantillons pour Parzen, graine aléatoire, découpage entraînement/validation. L'article ne précise que k = 1 et l'usage du momentum ; le reste est délégué au code. Dans la pratique, les fenêtres de Parzen de ce type de travaux sont ajustées sur 10 000 échantillons (Theis et al., 2016).

**Ressources.** Les auteurs remercient Compute Canada pour le calcul (p. 8). Les réseaux étant de petits perceptrons et MNIST de petite taille, nous pensons qu'un ordinateur ordinaire suffira. Un test d'entraînement d'un petit perceptron sur MNIST (une epoch, soit 60 000 images) prend environ 25 secondes sur notre ordinateur, sans GPU (processeur Intel Core i5-1135G7, 16 Go de RAM). Un GAN entraînant deux réseaux, nous estimons un coût de l'ordre de deux à quatre fois supérieur, soit une à deux minutes par epoch : cela reste faisable sur notre machine, et Google Colab pourra servir à accélérer les essais multiples.

## 8. Quelle est l'influence de l'article dans le temps ?

**Source.** Conférence NIPS 2014 (aujourd'hui NeurIPS), volume 27. Classement CORE : A\*.

**Citations (Google Scholar, relevé du 11/11/2026).**

- Total, versions fusionnées : **125 493** citations ; version NeurIPS 2014 : 100 712 (60 versions) ; version Communications of the ACM 2020 : 25 967.
- Tendance : forte croissance de 2016 à 2020, croissance plus lente ensuite avec un maximum en 2025 ; la barre de 2026 est plus basse, mais l'année n'est pas terminée et Scholar actualise avec retard, ce qui n'autorise pas à parler de déclin. (capture ci-dessous)
- Cohérence : environ 10 000 citations par an en moyenne depuis juin 2014, ce qui est cohérent avec un article fondateur dans un domaine très vaste et en forte croissance.

<figure style="margin:0; break-inside: avoid; page-break-inside: avoid;">
<img src="figures/scholar.png" alt="Capture Google Scholar" width="300">
<figcaption><em>Capture d'écran Google Scholar (relevé du 11/11/2026).</em></figcaption>
</figure>

**Qui cite et pourquoi** (nombres de citations : Google Scholar, 15.11.2026) :

- *Étend :* Radford et al. (arXiv 2015, DCGAN), cité 22 748 fois, remplacent les perceptrons par des réseaux convolutifs, une piste déjà amorcée par le modèle convolutif de la Figure 2d de l'article ; domaine : vision par ordinateur et apprentissage de représentations.
- *Critique :* Theis, van den Oord & Bethge (arXiv 2015, ICLR 2016), cité 1 590 fois, montrent que les estimations de Parzen sont peu fiables en grande dimension et qu'un simple modèle k-means peut battre la vraie distribution des données selon ce critère ; ils recommandent d'éviter cette métrique, celle-là même utilisée dans la Table 1. Domaine : évaluation des modèles génératifs.
- Domaines des citants en général : génération et traduction d'images, génération de séquences et de texte, publicité en ligne, prédiction vidéo, etc.

**Pertinence aujourd'hui.** Dhariwal & Nichol (NeurIPS 2021), cité 16 883 fois, montrent que les modèles de diffusion atteignent une meilleure qualité d'image que les meilleurs GAN de l'époque (FID de 2,97 sur ImageNet 128×128). Pour la génération d'images, les modèles de diffusion ont donc dépassé les GAN, mais l'idée du jeu adverse reste utilisée dans de nombreux domaines et l'article demeure une référence historique et conceptuelle majeure.

## 9. Réflexion : que pensons-nous de la qualité de l'écriture ?

Selon nous, l'article est globalement clair et bien structuré pour sa longueur (9 pages). L'analogie des faussaires et de la police (p. 1) et la Figure 1 (p. 4) rendent l'idée accessible, et la Table 2 (p. 8) offre un bon résumé comparatif. La partie la plus difficile est la section 4 (preuves) : elle suppose de connaître la divergence de Kullback-Leibler, la divergence de Jensen-Shannon et des notions de convexité, et la preuve de la Proposition 2 est très compacte. La difficulté vient donc surtout des connaissances préalables et non du style.

Quelques défauts d'écriture nuisent à la précision :

- L'Algorithme 1 (p. 4) minimise log(1 − D(G(z))), alors que le texte (p. 3) recommande de maximiser log D(G(z)).
- Le bruit y est tiré de « `p_g(z)` » alors que la notation définie est `p_z`.
- La preuve de la Proposition 2 conclut à la convergence vers « `p_x` » au lieu de `p_data`.
- Quelques coquilles (« indistiguishable », « succesful », « an a range of datasets ») et deux références identiques (les références 4 et 5).
- Les détails des expériences sont renvoyés au code plutôt que décrits dans le texte.

## Utilisation d'un modèle de langage (LLM)

Nous avons utilisé un modèle de langage (LLM) de trois façons. Dans un premier temps, pour nous aider à traduire les parties de l'article que nous ne comprenions pas, à cause de la langue et de termes mathématiques en anglais. Dans un second temps, pour nous aider dans nos recherches, par exemple pour trouver les textes qui citent notre article et pour comprendre des notions que nous ne connaissions pas. Enfin, il nous a été utile pour la mise en page et pour obtenir une rédaction propre. Il nous a aussi aidés à structurer le rapport et à rédiger des premiers brouillons de réponses aux questions, que nous avons ensuite relus, vérifiés avec l'article et adaptés.
