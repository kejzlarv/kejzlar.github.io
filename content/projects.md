+++
title = "Projects"
draft = false
+++

Here are projects that I have been working on during the past couple of years. They are a mix of Bayesian data analysis, machine learning modeling, and data science education. Take a look at the list of my [publications](/publications) and my [GitHub](https://github.com/kejzlarv) to see the fruits of this labor.

---

# Bayesian Mining of Nuclear Mass Data
{{< rawhtml >}}
<img style="float: Right; margin-left: 0.6em;"  src="/images/frib.jpg" width="400" height="270">
{{< /rawhtml >}}
The mass, or binding energy, is the basic property of the atomic nucleus. It determines nuclear stability as well as reaction and decay rates. Quantifying the nuclear binding is important for understanding the origin of the elements in the universe. I am one of the principal investigators in a team of researchers that strives to provide accurate values for atomic masses and their uncertainties beyond the range of available experimental data using physics-informed Bayesian machine learning. I have recently developed a novel approach to Bayesian stacking that utilizes Dirichlet distribution to leverage the collective wisdom of several theoretical nuclear mass models. Predictions obtained via this method show excellent performance on both prediction accuracy and uncertainty quantification (**21% improvement to the best theoretical model, 15% improvement to the best theoretical model corrected with Gaussian Process**).

---

# Uncertainty Quantification in Machine Learning
{{< rawhtml >}}
<img style="float: Left; margin-right: 0.6em;"  src="/images/UQ.png" width="400" height="270">
{{< /rawhtml >}}
I spend a lot of my time thinking about methods for efficient and accurate Uncertainty Quantification (UQ) in the context of machine learning models such as Gaussian processes or neural networks. UQ is a comprehensive study of the impact of all forms of modeling errors (“All models are wrong, but…,” you know how it goes). Take for instance the Global Forecast System (GFS), which is a numerical model for weather prediction developed by the National Centers for Environmental Protection. Although sophisticated, GFS is just a model that does not perfectly reflect the physical reality. GFS also depends on parameters that need to be estimated from data that are collected with a measurement error. Moreover, why should you look only at the GFS model and not consider other forecast systems? All of these considerations describe various sources of error that should be quantified in order to produce reliable inferences. Bayesian statistics has become a dominant UQ tool because it allows to express uncertainties intuitively in terms of probability. Despite its importance, carrying out even a simple UQ analysis in the context of machine learning models can be challenging due to the massive training datasets and model complexity. Over the past few years, I have developed various computationally fast and scalable methods for UQ in machine learning based on variational Bayesian inference that have surpassed the standard sampling-based inference.

---

# Data Science Education
{{< rawhtml >}}
<img style="float: right; margin-left: 0.6em;"  src="/images/AppletScreen.PNG" width="400" height="250">
{{< /rawhtml >}}
Probabilistic models such as logistic regression, Bayesian classification, neural networks, and models for natural language processing, are increasingly more present in both undergraduate and graduate statistics and data science curricula due to their wide range of applications. I create innovative resources focused on introducing realistic case studies in data science classrooms that facilitate statistical modeling and inference with large datasets.
