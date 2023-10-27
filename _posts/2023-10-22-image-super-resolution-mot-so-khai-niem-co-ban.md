---
title: '[Image super resolution] Một số khái niệm cơ bản'
date: 2023-10-22
permalink: /posts/2023/10/22/image-super-resolution-mot-so-khai-niem-co-ban/
tags:
  - super resolution
  - intelligent
---

Một số khái niệm cơ bản của mage super resolution và ứng dụng

Image super resolution
======

## Image super resolution (ISR) 

Nó là quá trình tăng cường độ phân giải của một hình ảnh, tức là làm cho hình ảnh trở nên rõ nét hơn bằng cách thêm thông tin chi tiết vào hình ảnh gốc. Điều này có thể hữu ích trong nhiều ứng dụng, bao gồm cải thiện chất lượng hình ảnh, nâng cao khả năng nhìn thấy chi tiết trong hình ảnh y khoa, và cải thiện khả năng nhận diện khuôn mặt hoặc đối tượng trong hình ảnh.

Cơ chế của ISR dựa trên Deep Learning:

1. **Convolutional Neural Networks (CNNs)**: Đây là một loại mạng nơ-ron thích hợp cho việc xử lý hình ảnh. CNNs sử dụng các tầng convolution để học các đặc trưng từ hình ảnh, từ các đặc trưng nhỏ như cạnh, góc, đến các đặc trưng lớn hơn như các đối tượng phức tạp.

2. **Deep Learning Architectures for Super Resolution**:
   - **Single Image Super Resolution (SISR)**: Dùng một hình ảnh đầu vào để tạo ra hình ảnh đầu ra với độ phân giải cao hơn. Mô hình sẽ học cách dự đoán các chi tiết bị mất đi trong quá trình giảm độ phân giải.
   
   - **Multi-Image Super Resolution (MISR)**: Sử dụng nhiều hình ảnh đầu vào với các góc chụp khác nhau để cải thiện độ phân giải. Điều này giúp cung cấp nhiều thông tin hơn để tái tạo các chi tiết bị mất đi.

3. **Loss Functions**: Các hàm mất mát (loss functions) được sử dụng để đo sự khác biệt giữa hình ảnh tái tạo và hình ảnh gốc. Phổ biến nhất là Mean Squared Error (MSE) hoặc Mean Absolute Error (MAE).

4. **Generative Adversarial Networks (GANs)**: Một phương pháp tiên tiến trong Deep Learning, GANs kết hợp cả một mô hình sinh (generator) và một mô hình phân biệt (discriminator) để cung cấp hình ảnh độ phân giải cao hơn và thậm chí chất lượng tốt hơn.


## Một số phương pháp thực hiện

Link source code for image super resolution: 

[Image super resolution - tutorial 01](https://github.com/JingyunLiang/SwinIR)

[Image super resolution - tutorial 02](https://github.com/xinntao/ESRGAN)

[Image super resolution - tutorial 03](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)

[Image super resolution - tutorial 04](https://github.com/IceClear/LDM-SRtuning/tree/main)

[Image super resolution - tutorial 05](https://github.com/ailabteam/zero123plus)

[Image super resolution - tutorial 06](https://github.com/ailabteam/Awesome-Video-Diffusion-Models)

[Image super resolution - tutorial 07](https://github.com/ailabteam/ScaleCrafter)


Hết.
