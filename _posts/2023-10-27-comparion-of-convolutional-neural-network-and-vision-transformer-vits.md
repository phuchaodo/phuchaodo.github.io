---
title: '[Note] Comparison of convolutional neural networks and vision transformers vits'
date: 2023-10-27
permalink: /posts/2023/10/27/comparion-of-convolutional-neural-network-and-vision-transformer-vits/
tags:
  - research
  - writing
  - transformers
  - neural networks
---

So sánh cơ chế, ưu và nhược điểm giữa CNN và vision transformer

Comparison of convolutional neural networks and vision transformers vits
======

## Some popular CNN architectures

That have had a significant impact on the field of computer vision include:

* LeNet-5: Developed by Yann LeCun and his colleagues in the 1990s, LeNet-5 is one of the earliest CNNs. It was initially designed for handwritten digit recognition and laid the foundation for future CNN architectures.

* AlexNet: Proposed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012, AlexNet achieved a breakthrough in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), dramatically outperforming traditional methods. It popularized the use of deeper and larger CNNs for image recognition tasks.

* VGG: Developed by the Visual Geometry Group at the University of Oxford in 2014, VGG demonstrated the benefits of using deeper networks with small (3x3) convolutional filters. Its simple and modular architecture made it a popular choice for transfer learning and fine-tuning in various applications.

* ResNet: Introduced by Kaiming He and his colleagues in 2015, ResNet addressed the issue of vanishing gradients in deep networks through the use of skip connections, allowing the network to learn residual functions. ResNet’s architecture enabled the training of much deeper networks, significantly improving performance on various computer vision benchmarks.

* DenseNet: Proposed by Gao Huang and his team in 2016, DenseNet introduced dense connections between layers, which alleviated the vanishing gradient problem and encouraged feature reuse. This led to more compact and efficient models compared to traditional CNNs.

These architectures, along with many others, have shaped the landscape of computer vision research and applications. They have demonstrated the power and versatility of CNNs, paving the way for new advancements in the field.


## Drawbacks of CNNs

While Convolutional Neural Networks (CNNs) have proven to be highly effective in various computer vision tasks, they are not without their limitations. Some of the main drawbacks of CNNs include:

* Requirement for Large Amounts of Labeled Data: CNNs typically rely on large, labeled datasets for training. The process of labeling and annotating image data can be time-consuming and expensive, making it a significant bottleneck in the development of new models. This also limits the applicability of CNNs in situations where labeled data is scarce or difficult to obtain.

* Computational Complexity: The deep and complex architectures of CNNs require substantial computational resources for both training and inference. This can pose challenges for deploying CNNs on resource-constrained devices, such as mobile phones and embedded systems, and may also increase the carbon footprint associated with training and running large-scale CNN models.

* Difficulties in Scaling up to High-Resolution Images: CNNs tend to struggle with scaling up to high-resolution images due to the increased memory and computational requirements associated with large input sizes. This has led to the development of various techniques to address this issue, such as using dilated convolutions or downsampling the input image. However, these approaches may compromise the model’s ability to capture fine-grained details in high-resolution images.

* Lack of Native Support for Variable-Sized Inputs: Traditional CNNs are designed to handle fixed-size input images, requiring input images to be resized or padded to match the network’s input dimensions. This can lead to distortion, loss of information, or the introduction of artifacts, potentially affecting the model’s performance. While some workarounds, such as fully convolutional networks (FCNs) or adaptive pooling layers, have been proposed to handle variable-sized inputs, native support for variable-sized inputs is not a standard feature in CNN architectures.

These limitations highlight some of the challenges associated with using CNNs in computer vision tasks and motivate the exploration of alternative architectures, such as Vision Transformers, which may offer improved performance and versatility in addressing these challenges.


## Vision Transformers (ViTs)

- Vision Transformers (ViTs) have emerged as a promising alternative to Convolutional Neural Networks (CNNs) for computer vision tasks. Initially designed for natural language processing, the Transformer architecture has demonstrated its versatility and effectiveness across various domains, including image recognition.

- General Concept of Vision Transformers: ViTs adapt the original Transformer architecture, which was initially proposed for sequence-to-sequence tasks in natural language processing, to handle images as input. Instead of processing images using convolutional operations, ViTs treat images as sequences of non-overlapping patches, converting each patch into a flat vector and applying positional encoding to retain spatial information. The resulting sequence of vectors is then fed into the Transformer layers, which utilize self-attention mechanisms to model long-range dependencies and learn meaningful visual features.

## How ViTs Work:

- Image Patching: The input image is divided into non-overlapping patches, typically of equal size, such as 16x16 or 32x32 pixels. Each patch is then reshaped into a 1D vector.

- Embedding: The reshaped patch vectors are linearly embedded into the desired dimensional space using a learnable embedding matrix.

- Positional Encoding: Positional embeddings are added to the patch embeddings to incorporate spatial information, ensuring that the Transformer model can distinguish between the different patch positions within the image.

- Transformer Layers: The combined patch and positional embeddings are passed through multiple Transformer layers, which consist of multi-head self-attention mechanisms and feed-forward neural networks, allowing the model to learn complex interactions between the patches.

- Output: The final output of the Transformer layers is processed by a classification head, typically a fully connected layer with a softmax activation, to generate the desired predictions, such as class probabilities for image classification tasks.

Popular ViT Architectures: Several ViT architectures have been proposed, with varying model sizes and configurations, to cater to different requirements and trade-offs between accuracy and computational efficiency. Some popular ViT architectures include:

* ViT-B: A base-sized ViT architecture with a moderate number of Transformer layers and attention heads, providing a good balance between performance and computational complexity.

* ViT-L: A large-sized ViT architecture with more Transformer layers and attention heads than ViT-B, offering improved performance at the cost of increased computational requirements.

* DeiT: Data-efficient Image Transformers (DeiT) are a family of ViTs designed to reduce the reliance on large-scale labeled data by incorporating techniques such as distillation and mixup during training, resulting in more data-efficient and generalizable models.

* ViTs have demonstrated their potential to outperform traditional CNNs on a variety of computer vision tasks, offering a new paradigm for visual recognition that leverages the strengths of the Transformer architecture.



## Performance Comparison: CNNs and ViTs

- When selecting the most appropriate architecture for computer vision tasks, it is crucial to compare the performance of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) in terms of classification accuracy and speed. In this section, we will discuss the performance differences between these two architectures.

- Classification Accuracy: Over the years, both CNNs and ViTs have achieved remarkable performance in image classification tasks. Initially, CNNs like AlexNet, VGG, ResNet, and DenseNet dominated the field by setting new benchmarks in terms of classification accuracy. However, with the introduction of ViTs, researchers have observed comparable or even superior performance in certain cases. ViTs have demonstrated state-of-the-art results on various image classification benchmarks, such as ImageNet, often outperforming their CNN counterparts. This improved accuracy can be attributed to the attention mechanism in ViTs, which allows them to capture long-range dependencies and contextual information more effectively.

- Computational Efficiency: CNNs have traditionally been known for their computational efficiency, particularly when compared to early deep learning models. However, ViTs have been shown to be competitive in this regard as well. While ViTs may require more computation during the initial stages of training, they often converge faster and require fewer epochs to achieve similar or better performance compared to CNNs. Furthermore, ViTs offer more architectural flexibility, making it easier to adapt the model to different computational budgets and problem sizes by adjusting the number of transformer layers or tokens.

- Inference Speed: In terms of inference speed, both architectures have their advantages and disadvantages. CNNs generally have a faster inference speed for smaller input sizes, whereas ViTs can process larger input images more efficiently due to their ability to attend to global information directly. However, the specific speed of each architecture can vary depending on factors such as hardware, implementation, and model size. Therefore, it is essential to consider the specific requirements of the application when comparing inference speeds.

In summary, both CNNs and ViTs exhibit strong performance in terms of classification accuracy and speed. While ViTs have shown promising results and in some cases outperform CNNs, it is important to recognize that each architecture has its strengths and weaknesses. Ultimately, the choice between CNNs and ViTs depends on the specific requirements of the application, including the desired classification accuracy, computational efficiency, and inference speed.


## Conclusion

- Throughout this article, we have examined the key differences between Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) in terms of their architecture, internal representations, robustness, and performance. The comparison highlights the unique strengths and weaknesses of both approaches, with ViTs emerging as a promising alternative to CNNs in various computer vision applications.

- ViTs have demonstrated superior performance in classification accuracy and robustness to adversarial perturbations in certain scenarios. They also exhibit distinct internal representation structures, characterized by consistent layer similarity and a combination of local and global attention across layers. However, ViTs are less inherently invariant to geometric transformations and may require data augmentation techniques during training to address this limitation.

- On the other hand, CNNs have been the backbone of computer vision tasks for years, offering computational efficiency and faster inference speeds for smaller input sizes. Their hierarchical representations and distinct layer similarity structures contribute to their robust performance in various tasks. Yet, CNNs face challenges in terms of scaling up to high-resolution images and require large amounts of labeled data for training.

In conclusion, the choice between CNNs and ViTs depends on the specific requirements of the application and the desired trade-offs between classification accuracy, robustness, and computational efficiency. Researchers and practitioners should carefully consider the unique properties of each architecture and adapt their models accordingly to achieve the best possible performance for their computer vision tasks. While the recent advancements in ViTs have shown great potential, it is important to recognize that both CNNs and ViTs will continue to play vital roles in the development of computer vision applications in the foreseeable future.


Hết.
