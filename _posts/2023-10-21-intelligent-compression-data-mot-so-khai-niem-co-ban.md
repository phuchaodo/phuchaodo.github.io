---
title: '[Intelligent data compression] Một số khái niệm cơ bản'
date: 2023-10-21
permalink: /posts/2023/10/21/intelligent-data-compression-mot-so-khai-niem-co-ban/
tags:
  - compression
  - intelligent
---

Một số khái niệm cơ bản của nén dữ liệu thông minh và ứng dụng

Nén dữ liệu thông minh
======

### Intelligent data compression (nén dữ liệu thông minh) 

Đây là quá trình sử dụng các thuật toán và kỹ thuật thông minh để nén dữ liệu một cách hiệu quả, giảm kích thước tệp tin mà vẫn đảm bảo sự bảo toàn và khôi phục dữ liệu một cách chính xác.

Các thành phần chính của intelligent data compression bao gồm:

* Thuật toán nén thông minh: Đây là phần quan trọng nhất của quá trình. Thuật toán nén thông minh sử dụng các phương pháp và kỹ thuật đặc biệt để tối ưu hóa việc giảm kích thước tệp tin mà không gây mất mát dữ liệu quan trọng.

* Kiểm soát mức độ nén: Intelligent compression cho phép người sử dụng điều chỉnh mức độ nén dựa trên ưu tiên cá nhân hoặc yêu cầu cụ thể. Điều này cho phép cân nhắc giữa việc giảm kích thước và bảo toàn chất lượng dữ liệu.

* Kiểm soát chất lượng và độ mất mát (trong trường hợp nén mất mát): Nếu đang sử dụng nén mất mát (lossy compression), intelligent compression cung cấp cách điều chỉnh mức độ mất mát dữ liệu. Người dùng có thể quyết định giữ lại mức chất lượng mong muốn.

* Hỗ trợ định dạng đa dạng: Intelligent compression thường hỗ trợ nén nhiều định dạng tệp tin, bao gồm hình ảnh, video, âm thanh, v.v.

* Tích hợp công nghệ tiên tiến: Các công nghệ như machine learning, deep learning và các phương pháp thông minh khác có thể được sử dụng để cải thiện quá trình nén dữ liệu.

* Kiểm soát mã hóa và bảo mật (tuỳ chọn): Intelligent compression có thể cung cấp các tùy chọn bảo mật để đảm bảo an toàn của dữ liệu sau khi được nén.


### Các loại nén dữ liệu thông minh

Intelligent data compression bao gồm cả lossy (nén mất mát) và lossless (nén không mất mát) compression.

#### 1. **Lossy Compression**:

   * Mất mát thông tin: Khi sử dụng lossy compression, một phần thông tin không quan trọng hoặc nhạy cảm với mắt người hoặc hệ thống có thể bị loại bỏ. Điều này dẫn đến việc giảm chất lượng của tệp nén so với tệp gốc.
   * Tỉ lệ nén cao: Lossy compression thường tạo ra các tệp nén có kích thước nhỏ hơn đáng kể so với tệp gốc. Điều này tiết kiệm không gian lưu trữ.
   * Sử dụng trong hình ảnh và video: Thường được sử dụng trong hình ảnh và video để giảm kích thước tệp mà vẫn duy trì mức độ chấp nhận được của chất lượng hình ảnh.

#### 2. **Lossless Compression**:

   * Không mất mát thông tin quan trọng: Lossless compression giữ nguyên toàn bộ thông tin của tệp gốc. Dữ liệu có thể được hoàn toàn phục hồi khi giải nén.
   * Tỉ lệ nén thấp hơn so với lossy: Lossless compression thường không tạo ra tỉ lệ nén lớn như lossy, vì nó không bỏ đi bất kỳ thông tin nào.
   * Sử dụng trong việc lưu trữ dữ liệu quan trọng: Thường được sử dụng khi việc duy trì chất lượng tuyệt đối của dữ liệu là ưu tiên cao, như trong y học, lưu trữ tệp tin không nén, và các trường hợp khác yêu cầu sự chính xác tuyệt đối.


#### Sử dụng **Lossy Compression**:

   * Việc giảm kích thước tệp quan trọng hơn so với việc giữ nguyên chất lượng tuyệt đối.
   * Dữ liệu có thể chấp nhận một mức độ mất mát thông tin.

#### Sử dụng **Lossless Compression**:
   
   * Việc duy trì chất lượng tuyệt đối của dữ liệu là quan trọng.
   * Dữ liệu yêu cầu sự chính xác tuyệt đối và không thể mất mát thông tin quan trọng.


Các thuật toán nén lossy
======

## 1. Variational Autoencoders (VAEs)

### Bước 1: Chuẩn bị dữ liệu
- Chuẩn bị dữ liệu đầu vào cần nén. Dữ liệu này sẽ được biểu diễn bằng một vector hoặc một tập hợp các vector.

### Bước 2: Xây dựng mô hình VAE

#### 2.1. **Encoder (Q)**:
- Nhận dữ liệu đầu vào và biến đổi thành hai giá trị: mean và log-variance (độ lệch chuẩn).
- Sử dụng một mạng nơ-ron để thực hiện quá trình này.

#### 2.2. **Decoder (P)**:
- Nhận một mẫu từ không gian tiếp cận (thông thường là không gian Gaussian) và giải mã nó thành một ước lượng của dữ liệu ban đầu.
- Như vậy, decoder hoạt động như một hàm phân phối xác suất.

#### 2.3. **Loss Function**:
- Gồm hai thành phần chính:
    - **Reconstruction Loss**: Đo sự sai khác giữa dữ liệu gốc và dữ liệu được giải mã từ không gian tiếp cận. Có thể sử dụng hàm cross-entropy hoặc MSE loss.
    - **Kullback-Leibler (KL) Loss**: Đo sự khác biệt giữa phân phối xác suất được sinh ra bởi encoder và một phân phối chuẩn.

#### 2.4. **Toàn bộ Loss Function**:
- Tổng hợp cả Reconstruction Loss và KL Loss.

### Bước 3: Huấn luyện mô hình

- Sử dụng tối ưu hóa gradient descent để tối ưu hóa loss function. 
- Mục tiêu là cực tiểu hóa tổng loss.

### Bước 4: Nén dữ liệu

- Sử dụng encoder để biến đổi dữ liệu đầu vào thành không gian tiếp cận. Dữ liệu trong không gian này thường có kích thước nhỏ hơn so với dữ liệu gốc.

### Bước 5: Khôi phục dữ liệu

- Sử dụng decoder để giải mã dữ liệu từ không gian tiếp cận trở lại dạng ban đầu.

Link source code VAEs cho lossy compression: [VAEs for lossy compression](https://github.com/duanzhiihao/lossy-vae)

Link source code DNN vs VAEs cho lossy compression: [DNN vs VAEs for lossy compression](https://github.com/NJUVISION/NIC/tree/main)


## 2. Generative Adversarial Networks (GANs)

Đây là một loại mô hình học sâu (deep learning) gồm hai thành phần chính: một mô hình sinh (generator) và một mô hình phân biệt (discriminator). GANs được huấn luyện thông qua quá trình cạnh tranh giữa hai mô hình này. Generator cố gắng tạo ra dữ liệu giống với dữ liệu thật từ một phân phối ẩn, trong khi discriminator cố gắng phân biệt giữa dữ liệu thật và dữ liệu được tạo ra bởi generator.

Ứng dụng GANs vào bài toán lossy compression đòi hỏi một số điều chỉnh và thêm một số thành phần:

### 1. **Dữ liệu đầu vào và đầu ra**:
   - Đầu vào: Ảnh, video hoặc dữ liệu nén khác.
   - Đầu ra: Ảnh, video hoặc dữ liệu nén có chất lượng thấp hơn.

### 2. **Mô hình Generator**:
   - Mô hình này sẽ cố gắng học cách biểu diễn và tạo ra dữ liệu nén.
   - Nếu áp dụng vào hình ảnh, đầu ra của generator có thể là các hình ảnh với độ phân giải thấp hơn.

### 3. **Mô hình Discriminator**:
   - Mô hình này sẽ học cách phân biệt giữa dữ liệu thật và dữ liệu được tạo ra bởi generator. Trong trường hợp này, dữ liệu thật sẽ là dữ liệu nén và dữ liệu giả là dữ liệu nén do generator tạo ra.

### 4. **Hàm mất mát**:
   - Mất mát của generator sẽ bao gồm hai thành phần:
      - Mất mát adversarial: Đây là một hàm mất mát binary cross-entropy giữa đầu ra của discriminator và các nhãn cho dữ liệu nén được tạo ra bởi generator (nhãn sẽ là 1, tương ứng với "dữ liệu thật").
      - Mất mát hồi quy (optional): Nếu cần, có thể thêm một hàm mất mát hồi quy để đảm bảo rằng dữ liệu nén sau khi giải nén có chất lượng tốt.

   - Mất mát của discriminator: Đây cũng là hàm mất mát binary cross-entropy giữa đầu ra của discriminator và các nhãn thật giả (nhãn sẽ là 1 cho dữ liệu thật và 0 cho dữ liệu giả).

### 5. **Quá trình huấn luyện**:
   - Trong mỗi vòng lặp, generator và discriminator sẽ được cập nhật tuần tự. Đầu tiên, generator tạo ra dữ liệu nén và cập nhật các trọng số để cố gắng "đánh lừa" discriminator. Sau đó, discriminator được huấn luyện để phân biệt giữa dữ liệu nén thật và giả.

### 6. **Cấu trúc mạng và siêu tham số**:
   - Đây là một phần quan trọng, và phụ thuộc vào bài toán cụ thể. Việc thiết kế generator và discriminator, cũng như lựa chọn siêu tham số (như learning rate) đòi hỏi sự thử nghiệm và điều chỉnh.

Link source code GAN cho lossy compression: [GAN for lossy compression](https://github.com/mit-han-lab/gan-compression)

## 3. Quá trình nén âm thanh lossy

Các bước chính của quá trình nén âm thanh lossy bao gồm:

1. **Phân tích âm thanh**:
   - Tín hiệu âm thanh ban đầu được phân tách thành các thành phần tần số khác nhau, thông qua biến đổi Fourier hoặc các phương pháp khác.

2. **Kỹ thuật mã hóa**:
   - Sử dụng các thuật toán nén để giảm dung lượng tệp âm thanh. Trong quá trình này, thông tin không cần thiết hoặc ít quan trọng bị loại bỏ.

3. **Lưu trữ tệp nén**:
   - Tệp âm thanh đã được nén sẽ có dung lượng nhỏ hơn và có thể được lưu trữ hoặc truyền đi nhanh hơn.

4. **Giải nén**:
   - Khi tệp nén được phát lại, nó sẽ được giải nén để tạo ra một bản sao gần giống với tín hiệu âm thanh ban đầu.

Link source code tutorial lossy compression: [Audio compression - tutorial 01](https://github.com/facebookresearch/audiocraft)

Link source code tutorial lossy compression: [Audio compression - tutorial 02](https://github.com/descriptinc/descript-audio-codec)


## 4. Learned Image Compression with Neural Networks

Nó là một phương pháp nén hình ảnh sử dụng mạng nơ-ron để học cách biểu diễn và nén hình ảnh. Điều này khác biệt với các phương pháp truyền thống dựa trên tiếp cận số học hoặc biến đổi tín hiệu.

* Cụ thể, LICN kết hợp một mô hình học sâu (neural network) để học một biểu diễn hiệu quả của hình ảnh. 
* Mạng nơ-ron này được huấn luyện trên một tập dữ liệu lớn các hình ảnh để học cách biểu diễn chúng trong một không gian dữ liệu thấp chiều hơn, giúp giảm thiểu dung lượng lưu trữ mà vẫn giữ lại lượng thông tin quan trọng.

Các thành phần quan trọng trong Learned Image Compression with Neural Networks bao gồm:

1. **Encoder Network**: Mạng này được sử dụng để biểu diễn hình ảnh ban đầu thành một biểu diễn tương đối nhỏ. Đây là phần của mô hình học sâu được huấn luyện để nén hình ảnh.

2. **Decoder Network**: Mạng này nhận vào biểu diễn nhỏ hơn và cố gắng khôi phục lại hình ảnh ban đầu từ đó. Nó chịu trách nhiệm giải nén dữ liệu và tạo ra hình ảnh gần giống với hình ảnh gốc.

3. **Loss Function**: Hàm mất mát định nghĩa cách đo sự sai lệch giữa hình ảnh gốc và hình ảnh được giải nén. Mục tiêu của mô hình là tối thiểu hóa hàm mất mát này để đảm bảo rằng hình ảnh giải nén sẽ gần giống hình ảnh gốc.

Công nghệ LICN cho phép một mức độ linh hoạt cao hơn trong việc tùy chỉnh việc nén hình ảnh theo yêu cầu cụ thể và cho phép điều chỉnh độ phân giải và chất lượng hình ảnh được giữ lại sau khi giải nén.

Link source code Learned Image Compression with Neural Networks: [Learned Image Compression with Neural Networks](https://github.com/jmliu206/LIC_TCM)


## 5. Video compression

Neural Network-Based Video Compression (NNBC) là một phương pháp nén video sử dụng mạng thần kinh nhân tạo (neural network) để giảm kích thước của dữ liệu video mà vẫn giữ được chất lượng hình ảnh tương đương. 

Dưới đây là một số thành phần chính của Neural Network-Based Video Compression:

1. **Encoder Network**: Mạng này chịu trách nhiệm chuyển đổi các khung hình video đầu vào thành một biểu diễn có kích thước nhỏ hơn, gọi là latent space. Mục tiêu của encoder là tìm cách biểu diễn dữ liệu video sao cho có thể tái tạo lại ảnh gốc từ biểu diễn này.

2. **Decoder Network**: Mạng này được sử dụng để giải nén biểu diễn nhỏ hơn (latent space) và tái tạo lại khung hình gốc. Mục tiêu của decoder là học cách biểu diễn các đặc trưng trong latent space để tái tạo lại ảnh ban đầu.

3. **Loss Function**: Hàm mất mát được sử dụng để đo lường sự sai lệch giữa ảnh gốc và ảnh tái tạo. Mục tiêu là tối thiểu hóa sự sai lệch này để đảm bảo chất lượng hình ảnh tái tạo là tốt nhất có thể.

4. **Training Data**: Để huấn luyện mạng, cần sử dụng một tập dữ liệu lớn gồm các video và các phiên bản đã được nén của chúng. Mạng sẽ học cách biểu diễn dữ liệu video sao cho có thể nén và giải nén lại mà không mất đi nhiều thông tin quan trọng.

5. **Quantization**: Kỹ thuật này thường được áp dụng để giới hạn số lượng giá trị mà mỗi phần tử trong biểu diễn có thể nhận. Điều này giúp giảm kích thước của dữ liệu biểu diễn.

6. **Entropy Coding**: Sau khi quantization, các giá trị được mã hóa bằng các phương pháp mã hóa thông tin như Huffman coding hoặc arithmetic coding để giảm kích thước tệp nén.

Lợi ích của Neural Network-Based Video Compression bao gồm:

- **Khả năng học cách biểu diễn dữ liệu phức tạp**: Mạng thần kinh có khả năng học cách biểu diễn các đặc trưng phức tạp của video, giúp nén dữ liệu một cách hiệu quả.

- **Giữ lại chất lượng tốt**: So với các phương pháp nén truyền thống, NNBC thường có khả năng giữ lại chất lượng hình ảnh tốt hơn ở mức độ nén cao.

- **Adaptability**: Mạng thần kinh có khả năng học từ dữ liệu mới, giúp cập nhật mô hình nén video khi có thay đổi về loại dữ liệu hoặc yêu cầu chất lượng nén.

Link source code Video compression: [Video compression - tutorial 01](https://github.com/flyywh/Image-compression-and-video-coding/tree/master)

Link source code Video compression: [Video compression - tutorial 02](https://github.com/ZhihaoHu/PyTorchVideoCompression)

Link source code Video compression: [Video compression - tutorial 03](https://github.com/mightydeveloper/Deep-Compression-PyTorch)


Các thuật toán nén lossless
======

## Giải thích các khái niệm và thuật toán đơn giản

Nén dữ liệu không mất mát (Lossless Compression) là quá trình giảm kích thước của dữ liệu mà không làm mất bất kỳ thông tin nào so với dữ liệu ban đầu. Đây là một quá trình khá đặc biệt, khác với nén mất mát (Lossy Compression) như JPEG hay MP3, nơi một phần thông tin có thể bị loại bỏ để giảm kích thước tệp.

Dưới đây là một số phương pháp đơn giản của nén dữ liệu không mất mát:

1. **Dictionary Coding (Huffman Coding)**:
   - **Ý tưởng**: Phương pháp này sử dụng một bảng từ điển (dictionary) để ánh xạ các biểu diễn thông tin với các mã ngắn nhất.
   - **Cách hoạt động**: Những phần dữ liệu thường xuất hiện nhiều lần trong tệp sẽ được biểu diễn bằng các mã ngắn hơn. Những phần dữ liệu ít phổ biến sẽ được biểu diễn bằng các mã dài hơn.

2. **Run-Length Encoding (RLE)**:
   - **Ý tưởng**: Kỹ thuật này nhóm các ký tự liên tiếp giống nhau thành một cặp gồm ký tự và số lượng lặp lại.
   - **Cách hoạt động**: Ví dụ, chuỗi "AAAABBBCCDAA" sẽ được biểu diễn thành "4A3B2C1D2A".

3. **Arithmetic Coding**:
   - **Ý tưởng**: Mỗi ký tự hoặc cụm ký tự trong dữ liệu đầu vào được ánh xạ vào một khoảng giá trị trên đoạn [0,1].
   - **Cách hoạt động**: Quá trình này liên quan đến việc chia nhỏ đoạn [0,1] thành các khoảng con tương ứng với xác suất xuất hiện của các ký tự hoặc cụm ký tự. Quá trình nén và giải nén sẽ sử dụng các khoảng này để tương tác với dữ liệu.

4. **Burrows-Wheeler Transform (BWT)**:
   - **Ý tưởng**: BWT tái sắp xếp các ký tự trong chuỗi để tạo ra các chuỗi có tính chất gom cụm, từ đó dễ dàng nén được.
   - **Cách hoạt động**: BWT sắp xếp lại các chuỗi thành các chuỗi tương tự nhưng chứa các cụm ký tự giống nhau. Sau đó, chuỗi đã biến đổi này thường được nén bằng RLE hoặc các kỹ thuật khác.

5. **Lempel-Ziv-Welch (LZW)**:
   - **Ý tưởng**: Kỹ thuật này xây dựng một bảng từ điển dựa trên chuỗi dữ liệu xuất hiện trong tệp tin.
   - **Cách hoạt động**: Chuỗi dữ liệu liên tiếp được so sánh với các phần tử trong bảng từ điển, và khi có sự trùng khớp, một mã được gán cho chuỗi đó và bảng từ điển được mở rộng.

Link source code for lossless compression: [Lossless compression - tutorial 01](https://github.com/kedartatwawadi/NN_compression)

Link source code for lossless compression: [Lossless compression - tutorial 02](https://github.com/Model-Compression/Lossless_Compression)

Link source code for lossless compression: [Lossless compression - tutorial 03](https://github.com/mohit1997/DeepZip)

Stable diffusion for image compression data
======

## Stable Diffusion Based Image Compression

Stable Diffusion makes for a very powerful lossy image compression codec.

Link source code SD for compression: [SD for compression - tutorial 01](https://pub.towardsai.net/stable-diffusion-based-image-compresssion-6f1f0a399202)

Link source code SD for compression: [SD for compression - tutorial 02](https://colab.research.google.com/drive/1Ci1VYHuFJK5eOX9TB0Mq4NsqkeDrMaaH?usp=sharing)

Link source code SD for compression: [SD for compression - tutorial 03](https://www.stavros.io/posts/compressing-images-with-stable-diffusion/)


Hết.
