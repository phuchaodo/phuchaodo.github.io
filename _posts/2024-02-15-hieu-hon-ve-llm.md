---
title: '[Note] Hiểu hơn về LLM'
date: 2024-02-15
permalink: /posts/2024/02/15/hieu-hon-ve-llm/
tags:
  - research
  - proposal
  - LLM
--- 

Hiểu hơn về LLM và cách xây dựng nó.



## 🧩 LLM Fundamentals

### 1. Mathematics for Machine Learning

Before mastering machine learning, it is important to understand the fundamental mathematical concepts that power these algorithms.

- **Linear Algebra**: This is crucial for understanding many algorithms, especially those used in deep learning. Key concepts include vectors, matrices, determinants, eigenvalues and eigenvectors, vector spaces, and linear transformations.
- **Calculus**: Many machine learning algorithms involve the optimization of continuous functions, which requires an understanding of derivatives, integrals, limits, and series. Multivariable calculus and the concept of gradients are also important.
- **Probability and Statistics**: These are crucial for understanding how models learn from data and make predictions. Key concepts include probability theory, random variables, probability distributions, expectations, variance, covariance, correlation, hypothesis testing, confidence intervals, maximum likelihood estimation, and Bayesian inference.


### 2. Python for Machine Learning

Python is a powerful and flexible programming language that's particularly good for machine learning, thanks to its readability, consistency, and robust ecosystem of data science libraries.

- **Python Basics**: Python programming requires a good understanding of the basic syntax, data types, error handling, and object-oriented programming.
- **Data Science Libraries**: It includes familiarity with NumPy for numerical operations, Pandas for data manipulation and analysis, Matplotlib and Seaborn for data visualization.
- **Data Preprocessing**: This involves feature scaling and normalization, handling missing data, outlier detection, categorical data encoding, and splitting data into training, validation, and test sets.
- **Machine Learning Libraries**: Proficiency with Scikit-learn, a library providing a wide selection of supervised and unsupervised learning algorithms, is vital. Understanding how to implement algorithms like linear regression, logistic regression, decision trees, random forests, k-nearest neighbors (K-NN), and K-means clustering is important. Dimensionality reduction techniques like PCA and t-SNE are also helpful for visualizing high-dimensional data.

### 3. Neural Networks

Neural networks are a fundamental part of many machine learning models, particularly in the realm of deep learning. To utilize them effectively, a comprehensive understanding of their design and mechanics is essential.

- **Fundamentals**: This includes understanding the structure of a neural network such as layers, weights, biases, and activation functions (sigmoid, tanh, ReLU, etc.)
- **Training and Optimization**: Familiarize yourself with backpropagation and different types of loss functions, like Mean Squared Error (MSE) and Cross-Entropy. Understand various optimization algorithms like Gradient Descent, Stochastic Gradient Descent, RMSprop, and Adam.
- **Overfitting**: Understand the concept of overfitting (where a model performs well on training data but poorly on unseen data) and learn various regularization techniques (dropout, L1/L2 regularization, early stopping, data augmentation) to prevent it.
- **Implement a Multilayer Perceptron (MLP)**: Build an MLP, also known as a fully connected network, using PyTorch.


### 4. Natural Language Processing (NLP)

NLP is a fascinating branch of artificial intelligence that bridges the gap between human language and machine understanding. From simple text processing to understanding linguistic nuances, NLP plays a crucial role in many applications like translation, sentiment analysis, chatbots, and much more.

- **Text Preprocessing**: Learn various text preprocessing steps like tokenization (splitting text into words or sentences), stemming (reducing words to their root form), lemmatization (similar to stemming but considers the context), stop word removal, etc.
- **Feature Extraction Techniques**: Become familiar with techniques to convert text data into a format that can be understood by machine learning algorithms. Key methods include Bag-of-words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), and n-grams.
- **Word Embeddings**: Word embeddings are a type of word representation that allows words with similar meanings to have similar representations. Key methods include Word2Vec, GloVe, and FastText.
- **Recurrent Neural Networks (RNNs)**: Understand the working of RNNs, a type of neural network designed to work with sequence data. Explore LSTMs and GRUs, two RNN variants that are capable of learning long-term dependencies.


## 🧑‍🔬 The LLM Scientist

This section of the course focuses on learning how to build the best possible LLMs using the latest techniques.

### 1. The LLM architecture

While an in-depth knowledge about the Transformer architecture is not required, it is important to have a good understanding of its inputs (tokens) and outputs (logits). The vanilla attention mechanism is another crucial component to master, as improved versions of it are introduced later on.

* **High-level view**: Revisit the encoder-decoder Transformer architecture, and more specifically the decoder-only GPT architecture, which is used in every modern LLM.
* **Tokenization**: Understand how to convert raw text data into a format that the model can understand, which involves splitting the text into tokens (usually words or subwords).
* **Attention mechanisms**: Grasp the theory behind attention mechanisms, including self-attention and scaled dot-product attention, which allows the model to focus on different parts of the input when producing an output.
* **Text generation**: Learn about the different ways the model can generate output sequences. Common strategies include greedy decoding, beam search, top-k sampling, and nucleus sampling.


### 2. Building an instruction dataset

While it's easy to find raw data from Wikipedia and other websites, it's difficult to collect pairs of instructions and answers in the wild. Like in traditional machine learning, the quality of the dataset will directly influence the quality of the model, which is why it might be the most important component in the fine-tuning process.

* **[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)-like dataset**: Generate synthetic data from scratch with the OpenAI API (GPT). You can specify seeds and system prompts to create a diverse dataset.
* **Advanced techniques**: Learn how to improve existing datasets with [Evol-Instruct](https://arxiv.org/abs/2304.12244), how to generate high-quality synthetic data like in the [Orca](https://arxiv.org/abs/2306.02707) and [phi-1](https://arxiv.org/abs/2306.11644) papers.
* **Filtering data**: Traditional techniques involving regex, removing near-duplicates, focusing on answers with a high number of tokens, etc.
* **Prompt templates**: There's no true standard way of formatting instructions and answers, which is why it's important to know about the different chat templates, such as [ChatML](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python&pivots=programming-language-chat-ml), [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), etc.


### 3. Pre-training models

Pre-training is a very long and costly process, which is why this is not the focus of this course. It's good to have some level of understanding of what happens during pre-training, but hands-on experience is not required.

* **Data pipeline**: Pre-training requires huge datasets (e.g., [Llama 2](https://arxiv.org/abs/2307.09288) was trained on 2 trillion tokens) that need to be filtered, tokenized, and collated with a pre-defined vocabulary.
* **Causal language modeling**: Learn the difference between causal and masked language modeling, as well as the loss function used in this case. For efficient pre-training, learn more about [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) or [gpt-neox](https://github.com/EleutherAI/gpt-neox).
* **Scaling laws**: The [scaling laws](https://arxiv.org/pdf/2001.08361.pdf) describe the expected model performance based on the model size, dataset size, and the amount of compute used for training.
* **High-Performance Computing**: Out of scope here, but more knowledge about HPC is fundamental if you're planning to create your own LLM from scratch (hardware, distributed workload, etc.).


### 4. Supervised Fine-Tuning

Pre-trained models are only trained on a next-token prediction task, which is why they're not helpful assistants. SFT allows you to tweak them to respond to instructions. Moreover, it allows you to fine-tune your model on any data (private, not seen by GPT-4, etc.) and use it without having to pay for an API like OpenAI's.

* **Full fine-tuning**: Full fine-tuning refers to training all the parameters in the model. It is not an efficient technique, but it produces slightly better results.
* [**LoRA**](https://arxiv.org/abs/2106.09685): A parameter-efficient technique (PEFT) based on low-rank adapters. Instead of training all the parameters, we only train these adapters.
* [**QLoRA**](https://arxiv.org/abs/2305.14314): Another PEFT based on LoRA, which also quantizes the weights of the model in 4 bits and introduce paged optimizers to manage memory spikes. Combine it with [Unsloth](https://github.com/unslothai/unsloth) to run it efficiently on a free Colab notebook.
* **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: A user-friendly and powerful fine-tuning tool that is used in a lot of state-of-the-art open-source models.
* [**DeepSpeed**](https://www.deepspeed.ai/): Efficient pre-training and fine-tuning of LLMs for multi-GPU and multi-node settings (implemented in Axolotl).


### 5. Reinforcement Learning from Human Feedback

After supervised fine-tuning, RLHF is a step used to align the LLM's answers with human expectations. The idea is to learn preferences from human (or artificial) feedback, which can be used to reduce biases, censor models, or make them act in a more useful way. It is more complex than SFT and often seen as optional.

* **Preference datasets**: These datasets typically contain several answers with some kind of ranking, which makes them more difficult to produce than instruction datasets.
* [**Proximal Policy Optimization**](https://arxiv.org/abs/1707.06347): This algorithm leverages a reward model that predicts whether a given text is highly ranked by humans. This prediction is then used to optimize the SFT model with a penalty based on KL divergence.
* **[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)**: DPO simplifies the process by reframing it as a classification problem. It uses a reference model instead of a reward model (no training needed) and only requires one hyperparameter, making it more stable and efficient.


### 6. Evaluation

Evaluating LLMs is an undervalued part of the pipeline, which is time-consuming and moderately reliable. Your downstream task should dictate what you want to evaluate, but always remember Goodhart's law: "When a measure becomes a target, it ceases to be a good measure."

* **Traditional metrics**: Metrics like perplexity and BLEU score are not as popular as they were because they're flawed in most contexts. It is still important to understand them and when they can be applied.
* **General benchmarks**: Based on the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) is the main benchmark for general-purpose LLMs (like ChatGPT). There are other popular benchmarks like [BigBench](https://github.com/google/BIG-bench), [MT-Bench](https://arxiv.org/abs/2306.05685), etc.
* **Task-specific benchmarks**: Tasks like summarization, translation, and question answering have dedicated benchmarks, metrics, and even subdomains (medical, financial, etc.), such as [PubMedQA](https://pubmedqa.github.io/) for biomedical question answering.
* **Human evaluation**: The most reliable evaluation is the acceptance rate by users or comparisons made by humans. If you want to know if a model performs well, the simplest but surest way is to use it yourself.


### 7. Quantization

Quantization is the process of converting the weights (and activations) of a model using a lower precision. For example, weights stored using 16 bits can be converted into a 4-bit representation. This technique has become increasingly important to reduce the computational and memory costs associated with LLMs.

* **Base techniques**: Learn the different levels of precision (FP32, FP16, INT8, etc.) and how to perform naïve quantization with absmax and zero-point techniques.
* **GGUF and llama.cpp**: Originally designed to run on CPUs, [llama.cpp](https://github.com/ggerganov/llama.cpp) and the GGUF format have become the most popular tools to run LLMs on consumer-grade hardware.
* **GPTQ and EXL2**: [GPTQ](https://arxiv.org/abs/2210.17323) and, more specifically, the [EXL2](https://github.com/turboderp/exllamav2) format offer an incredible speed but can only run on GPUs. Models also take a long time to be quantized.
* **AWQ**: This new format is more accurate than GPTQ (lower perplexity) but uses a lot more VRAM and is not necessarily faster.


### 8. New Trends

* **Positional embeddings**: Learn how LLMs encode positions, especially relative positional encoding schemes like [RoPE](https://arxiv.org/abs/2104.09864). Implement [YaRN](https://arxiv.org/abs/2309.00071) (multiplies the attention matrix by a temperature factor) or [ALiBi](https://arxiv.org/abs/2108.12409) (attention penalty based on token distance) to extend the context length.
* **Model merging**: Merging trained models has become a popular way of creating peformant models without any fine-tuning. The popular [mergekit](https://github.com/cg123/mergekit) library implements the most popular merging methods, like SLERP, [DARE](https://arxiv.org/abs/2311.03099), and [TIES](https://arxiv.org/abs/2311.03099).
* **Mixture of Experts**: [Mixtral](https://arxiv.org/abs/2401.04088) re-popularized the MoE architecture thanks to its excellent performance. In parallel, a type of frankenMoE emerged in the OSS community by merging models like [Phixtral](https://huggingface.co/mlabonne/phixtral-2x2_8), which is a cheaper and performant option.
* **Multimodal models**: These models (like [CLIP](https://openai.com/research/clip), [Stable Diffusion](https://stability.ai/stable-image), or [LLaVA](https://llava-vl.github.io/)) process multiple types of inputs (text, images, audio, etc.) with a unified embedding space, which unlocks powerful applications like text-to-image.


## 👷 The LLM Engineer

This section of the course focuses on learning how to build LLM-powered applications that can be used in production, with a focus on augmenting models and deploying them.


### 1. Running LLMs

Running LLMs can be difficult due to high hardware requirements. Depending on your use case, you might want to simply consume a model through an API (like GPT-4) or run it locally. In any case, additional prompting and guidance techniques can improve and constrain the output for your applications.

* **LLM APIs**: APIs are a convenient way to deploy LLMs. This space is divided between private LLMs ([OpenAI](https://platform.openai.com/), [Google](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview), [Anthropic](https://docs.anthropic.com/claude/reference/getting-started-with-the-api), [Cohere](https://docs.cohere.com/docs), etc.) and open-source LLMs ([OpenRouter](https://openrouter.ai/), [Hugging Face](https://huggingface.co/inference-api), [Together AI](https://www.together.ai/), etc.).
* **Open-source LLMs**: The [Hugging Face Hub](https://huggingface.co/models) is a great place to find LLMs. You can directly run some of them in [Hugging Face Spaces](https://huggingface.co/spaces), or download and run them locally in apps like [LM Studio](https://lmstudio.ai/) or through the CLI with [llama.cpp](https://github.com/ggerganov/llama.cpp) or [Ollama](https://ollama.ai/).
* **Prompt engineering**: Common techniques include zero-shot prompting, few-shot prompting, chain of thought, and ReAct. They work better with bigger models, but can be adapted to smaller ones.
* **Structuring outputs**: Many tasks require a structured output, like a strict template or a JSON format. Libraries like [LMQL](https://lmql.ai/), [Outlines](https://github.com/outlines-dev/outlines), [Guidance](https://github.com/guidance-ai/guidance), etc. can be used to guide the generation and respect a given structure.


### 2. Building a Vector Storage

Creating a vector storage is the first step to build a Retrieval Augmented Generation (RAG) pipeline. Documents are loaded, split, and relevant chunks are used to produce vector representations (embeddings) that are stored for future use during inference.

* **Ingesting documents**: Document loaders are convenient wrappers that can handle many formats: PDF, JSON, HTML, Markdown, etc. They can also directly retrieve data from some databases and APIs (GitHub, Reddit, Google Drive, etc.).
* **Splitting documents**: Text splitters break down documents into smaller, semantically meaningful chunks. Instead of splitting text after *n* characters, it's often better to split by header or recursively, with some additional metadata.
* **Embedding models**: Embedding models convert text into vector representations. It allows for a deeper and more nuanced understanding of language, which is essential to perform semantic search.
* **Vector databases**: Vector databases (like [Chroma](https://www.trychroma.com/), [Pinecone](https://www.pinecone.io/), [Milvus](https://milvus.io/), [FAISS](https://faiss.ai/), [Annoy](https://github.com/spotify/annoy), etc.) are designed to store embedding vectors. They enable efficient retrieval of data that is 'most similar' to a query based on vector similarity.


### 3. Retrieval Augmented Generation

With RAG, LLMs retrieves contextual documents from a database to improve the accuracy of their answers. RAG is a popular way of augmenting the model's knowledge without any fine-tuning.

* **Orchestrators**: Orchestrators (like [LangChain](https://python.langchain.com/docs/get_started/introduction), [LlamaIndex](https://docs.llamaindex.ai/en/stable/), [FastRAG](https://github.com/IntelLabs/fastRAG), etc.) are popular frameworks to connect your LLMs with tools, databases, memories, etc. and augment their abilities.
* **Retrievers**: User instructions are not optimized for retrieval. Different techniques (e.g., multi-query retriever, [HyDE](https://arxiv.org/abs/2212.10496), etc.) can be applied to rephrase/expand them and improve performance.
* **Memory**: To remember previous instructions and answers, LLMs and chatbots like ChatGPT add this history to their context window. This buffer can be improved with summarization (e.g., using a smaller LLM), a vector store + RAG, etc.
* **Evaluation**: We need to evaluate both the document retrieval (context precision and recall) and generation stages (faithfulness and answer relevancy). It can be simplified with tools [Ragas](https://github.com/explodinggradients/ragas/tree/main) and [DeepEval](https://github.com/confident-ai/deepeval).


### 4. Advanced RAG

Real-life applications can require complex pipelines, including SQL or graph databases, as well as automatically selecting relevant tools and APIs. These advanced techniques can improve a baseline solution and provide additional features.

* **Query construction**: Structured data stored in traditional databases requires a specific query language like SQL, Cypher, metadata, etc. We can directly translate the user instruction into a query to access the data with query construction.
* **Agents and tools**: Agents augment LLMs by automatically selecting the most relevant tools to provide an answer. These tools can be as simple as using Google or Wikipedia, or more complex like a Python interpreter or Jira. 
* **Post-processing**: Final step that processes the inputs that are fed to the LLM. It enhances the relevance and diversity of documents retrieved with re-ranking, [RAG-fusion](https://github.com/Raudaschl/rag-fusion), and classification.


### 5. Inference optimization

Text generation is a costly process that requires expensive hardware. In addition to quantization, various techniques have been proposed to maximize throughput and reduce inference costs.

* **Flash Attention**: Optimization of the attention mechanism to transform its complexity from quadratic to linear, speeding up both training and inference.
* **Key-value cache**: Understand the key-value cache and the improvements introduced in [Multi-Query Attention](https://arxiv.org/abs/1911.02150) (MQA) and [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) (GQA).
* **Speculative decoding**: Use a small model to produce drafts that are then reviewed by a larger model to speed up text generation.


### 6. Deploying LLMs

Deploying LLMs at scale is an engineering feat that can require multiple clusters of GPUs. In other scenarios, demos and local apps can be achieved with a much lower complexity. 

* **Local deployment**: Privacy is an important advantage that open-source LLMs have over private ones. Local LLM servers ([LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/), [oobabooga](https://github.com/oobabooga/text-generation-webui), [kobold.cpp](https://github.com/LostRuins/koboldcpp), etc.) capitalize on this advantage to power local apps. 
* **Demo deployment**: Frameworks like [Gradio](https://www.gradio.app/) and [Streamlit](https://docs.streamlit.io/) are helpful to prototype applications and share demos. You can also easily host them online, for example using [Hugging Face Spaces](https://huggingface.co/spaces).
* **Server deployment**: Deploy LLMs at scale requires cloud (see also [SkyPilot](https://skypilot.readthedocs.io/en/latest/)) or on-prem infrastructure and often leverage optimized text generation frameworks like [TGI](https://github.com/huggingface/text-generation-inference), [vLLM](https://github.com/vllm-project/vllm/tree/main), etc.
* **Edge deployment**: In constrained environments, high-performance frameworks like [MLC LLM](https://github.com/mlc-ai/mlc-llm) and [mnn-llm](https://github.com/wangzhaode/mnn-llm/blob/master/README_en.md) can deploy LLM in web browsers, Android, and iOS.

### 7. Securing LLMs

In addition to traditional security problems associated with software, LLMs have unique weaknesses due to the way they are trained and prompted.

* **Prompt hacking**: Different techniques related to prompt engineering, including prompt injection (additional instruction to hijack the model's answer), data/prompt leaking (retrieve its original data/prompt), and jailbreaking (craft prompts to bypass safety features).
* **Backdoors**: Attack vectors can target the training data itself, by poisoning the training data (e.g., with false information) or creating backdoors (secret triggers to change the model's behavior during inference).
* **Defensive measures**: The best way to protect your LLM applications is to test them against these vulnerabilities (e.g., using red teaming and checks like [garak](https://github.com/leondz/garak/)) and observe them in production (with a framework like [langfuse](https://github.com/langfuse/langfuse)).


Link ref: 

[Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)

[Link 1](https://github.com/simonw/llm)

[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

[LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)

[llm-course](https://github.com/mlabonne/llm-course)


Hết.
