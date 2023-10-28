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

Link source code for image super resolution.

Sử dụng GAN model

[Image super resolution - tutorial 01](https://github.com/JingyunLiang/SwinIR)

[Image super resolution - tutorial 02](https://github.com/xinntao/ESRGAN)

Sử dụng diffusion model

[Image super resolution - tutorial 03](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)

[Image super resolution - tutorial 04](https://github.com/IceClear/LDM-SRtuning/tree/main)

[Image super resolution - tutorial 05](https://github.com/ailabteam/zero123plus)

[Image super resolution - tutorial 06](https://github.com/ailabteam/Awesome-Video-Diffusion-Models)

[Image super resolution - tutorial 07](https://github.com/ailabteam/ScaleCrafter)

Cơ chế hoạt động của một số mô hình
======

## Mô hình diffusion (Diffusion Model)

Nó là một phương pháp trong lĩnh vực Image Super-Resolution (ISR) sử dụng để tăng độ phân giải của hình ảnh. Cơ chế này sử dụng sự kết hợp giữa mô hình diffusion và deep learning để đạt được kết quả tốt.

Dưới đây là cách cơ chế sử dụng mô hình diffusion vào việc thực hiện ISR:

1. **Mô hình Diffusion**:
   - Diffusion Model là một mô hình xác suất thống kê, nó mô tả cách dữ liệu thay đổi theo thời gian thông qua các bước nhỏ. Trong ngữ cảnh này, nó có thể được sử dụng để tạo ra các phiên bản có độ phân giải cao hơn của hình ảnh.
   - Phương pháp này đưa ra giả định rằng ảnh độ phân giải cao có thể được tạo ra bằng cách tiến hành nhiều bước nhỏ từ ảnh độ phân giải thấp.

2. **Kết hợp với Deep Learning**:
   - Một mạng học sâu, thường là một mạng neural convolutional (CNN), được sử dụng để học mối quan hệ phức tạp giữa các phiên bản ảnh ở các mức độ phân giải khác nhau.
   - Mô hình học sâu sẽ học cách chuyển đổi ảnh thấp độ phân giải thành các biến thể ảnh có độ phân giải cao hơn, sử dụng thông tin từ mô hình diffusion.

3. **Quá trình tiên đoán**:
   - Khi mô hình đã được huấn luyện, nó có thể được sử dụng để tăng độ phân giải của các hình ảnh mới.
   - Quá trình này giúp tạo ra các phiên bản ảnh có độ phân giải cao hơn từ ảnh gốc độ phân giải thấp.


## Mô hình Generative Adversarial Network (GAN)

Nó cũng có thể được sử dụng trong việc thực hiện Image Super-Resolution (ISR). GANs là một kiến trúc mạng neural đặc biệt, bao gồm hai phần: một mạng sinh (generator) và một mạng phân biệt (discriminator). Các bước cơ bản để sử dụng GAN model trong ISR bao gồm:

1. **Generator (Mạng Sinh)**:
   - Generator nhận đầu vào là ảnh có độ phân giải thấp và cố gắng sinh ra một ảnh với độ phân giải cao hơn. 
   - Trong bối cảnh ISR, mục tiêu của Generator là tạo ra một ảnh cao độ phân giải từ ảnh thấp độ phân giải.

2. **Discriminator (Mạng Phân Biệt)**:
   - Discriminator cố gắng phân biệt giữa ảnh thật (có độ phân giải cao) và ảnh giả (được tạo ra bởi Generator).
   - Nhiệm vụ của Discriminator là học cách phân biệt giữa ảnh cao độ phân giải và ảnh được sinh ra.

3. **Huấn luyện GAN**:
   - GAN được huấn luyện thông qua việc tối ưu hóa hai mục tiêu:
     - **Generator Loss**: Đảm bảo rằng ảnh được sinh ra bởi Generator gần giống với ảnh cao độ phân giải thật. Điều này được đo bằng cách sử dụng một hàm mất mát, thường là Mean Squared Error (MSE) hoặc hàm mất mát L1.
     - **Discriminator Loss**: Đảm bảo rằng Discriminator phân biệt đúng giữa ảnh thật và ảnh giả. Hàm mất mát này thúc đẩy Discriminator học cách phân biệt tốt hơn.

4. **Quá trình tiên đoán**:
   - Sau khi GAN đã được huấn luyện, nó có thể được sử dụng để tăng độ phân giải của các hình ảnh mới.
   - Generator nhận đầu vào là ảnh thấp độ phân giải và tạo ra một ảnh cao độ phân giải.

## Một số phương pháp khác

Có nhiều phương pháp khác để thực hiện Image Super-Resolution (ISR), ngoài phương pháp diffusion và sử dụng mô hình Generative Adversarial Network (GAN). Dưới đây là một số phương pháp khác:

1. **Interpolation-based methods**:
   - Phương pháp nội suy: Sử dụng các kỹ thuật nội suy như bilinear, bicubic để tăng độ phân giải của ảnh. Đây là các phương pháp đơn giản và nhanh chóng, nhưng không thể tạo ra các chi tiết mới hoặc cải thiện chất lượng ảnh.

2. **Sparse-coding-based methods**:
   - Sử dụng các kỹ thuật như sparse coding để học các bộ cơ sở từ dữ liệu huấn luyện và sử dụng chúng để tăng độ phân giải.

3. **Edge-based methods**:
   - Sử dụng thông tin biên để tăng độ phân giải. Các phương pháp này tập trung vào việc cải thiện các đường biên và các cạnh của đối tượng trong hình ảnh.

4. **Deep learning-based methods (non-GAN)**:
   - Sử dụng các mô hình học sâu khác ngoài GAN, như Convolutional Neural Networks (CNNs), để học mối quan hệ giữa các phiên bản ảnh ở các mức độ phân giải khác nhau.

5. **Self-examples-based methods**:
   - Sử dụng các ví dụ tự tham chiếu từ cùng một hình ảnh để tạo ra các phiên bản có độ phân giải cao hơn.

6. **Enhancement-based methods**:
   - Tập trung vào việc cải thiện các phần cụ thể của ảnh, chẳng hạn như việc tăng cường các đường biên hoặc cải thiện sự tương phản.


Hết.
