---
title: '[Note] Kiến thức cơ bản về xác suất thống kê'
date: 2024-02-05
permalink: /posts/2024/02/05/kien-thuc-co-ban-xac-suat-thong-ke/
tags:
  - paper
  - writing
  - science
  - thongke
---

Kiến thức cơ bản về xác suất thống kê

Statistics
======

Statistics is the foundation of data science. It provides the tools and methods needed to collect, analyze, and interpret data in a meaningful way. 

Here are some of the most common statistical methods used in data science:

* **Descriptive statistics:** Mean, median, mode, standard deviation, variance, percentiles, histograms, boxplots, scatterplots
* **Inferential statistics:** Hypothesis testing, confidence intervals, p-values, t-tests, ANOVA, chi-square tests
* **Regression analysis:** Simple linear regression, multiple linear regression, logistic regression
* **Time series analysis:** Autocorrelation, ARIMA models
* **Classification:** Decision trees, random forests, support vector machines
* **Clustering:** K-means clustering, hierarchical clustering


Sampling techniques
======

Sampling techniques are crucial in data science as they allow you to analyze smaller, manageable subsets (samples) of data to draw inferences about the larger population you're interested in. 
Choosing the right technique depends on your specific goals and data characteristics. Here's an overview of the main categories:

**Probability Sampling:**

* **Every member of the population has a known chance of being selected.** This guarantees representativeness and allows for statistical inferences about the population.
    * **Simple Random Sampling:** Each member has an equal chance, often achieved through random number generation. Good for homogeneous populations.
    * **Stratified Sampling:** Divide the population into subgroups (strata) based on shared characteristics and then randomly sample from each stratum proportionally. Ensures representation of diverse subgroups.
    * **Systematic Sampling:** Select members at fixed intervals from a pre-ordered list. Efficient but sensitive to ordering bias.
    * **Cluster Sampling:** Group the population into clusters, randomly select some clusters, and then include all members from those clusters. Useful when individual members are difficult to access.

**Non-Probability Sampling:**

* **Selection is not based on random chance.** Often used for exploratory research or when obtaining a random sample is impractical.
    * **Convenience Sampling:** Select readily available members, like volunteers or online participants. Quick and easy but not representative.
    * **Purposive Sampling:** Select members based on specific characteristics relevant to your research question. Useful for in-depth understanding of specific groups.
    * **Snowball Sampling:** Ask initial participants to identify others who fit your criteria. Useful for rare populations but can lead to biased samples.
    * **Quota Sampling:** Set quotas for different subgroups to ensure some level of representativeness but still not random.

**Additional factors to consider:**

* **Sample size:** Larger samples generally lead to more accurate inferences, but diminishing returns occur.
* **Sampling error:** The inherent difference between your sample and the population, unavoidable but can be estimated.
* **Ethical considerations:** Ensure your sampling method respects privacy and avoids discrimination.


Probability Distributions
======

Data Types: - we have Qualitative and Quantitative data. And in Quantitative  data, we have Continuous and Discrete data types. 
➢ Continuous data is measured and can take any number of values in a given finite or infinite range. It can be represented in decimal format. 
And the random variable that holds continuous values is called the Continuous  random variable. 
Examples: A person’s height, Time, distance, etc. 

➢ Discrete data is counted and can take only a limited number of values. It  makes no sense when written in decimal format. And the random variable  that holds discrete data is called the Discrete random variable. 
Example: The number of students in a class, number of workers in a  company, etc. 

Types of Probability Distributions 

Two major kinds of distributions based on the type of likely values for the  variables are, 
1. Discrete Distributions 
2. Continuous Distributions 

Different type of distribution of data: - 

i. Bernoulli Distribution 

ii. Uniform Distribution 

iii. Binomial Distribution 

iv. Normal or Gaussian Distribution 

v. Exponential Distribution 

vi. Poisson Distribution 


Note: nên phân tích kỹ hơn các khái niệm (đã được đề cập ở trên)


Hết.
