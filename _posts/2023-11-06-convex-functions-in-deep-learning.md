---
title: '[Note] Convex Functions in Deep Learning'
date: 2023-11-06
permalink: /posts/2023/11/06/convex-functions-in-deep-learning/
tags:
  - research
  - writing
  - convex
  - function
--- 

Khái niệm về hàm lồi và ứng dụng trong deep learning

Khái niệm về hàm lồi
======

Một hàm lồi (convex function) trong toán học là một hàm số mà đường tiếp tuyến của bất kỳ hai điểm nào trên đồ thị của nó không bao giờ cắt qua đồ thị của nó. Nói cách khác, một hàm được gọi là lồi nếu đoạn thẳng kết nối hai điểm bất kỳ trên đồ thị của nó không bao giờ nằm dưới đồ thị đó.

Ứng dụng của hàm lồi rất phong phú trong nhiều lĩnh vực khác nhau của toán học và khoa học máy tính. Dưới đây là một số ví dụ:

1. **Tối ưu hóa**: Hàm lồi rất quan trọng trong tối ưu hóa. Bản chất lồi của hàm giúp đảm bảo rằng một điểm cực tiểu cục bộ cũng chính là điểm cực tiểu toàn cục, điều này giúp giải quyết các vấn đề tối ưu hóa một cách hiệu quả.

2. **Học máy và học sâu (deep learning)**: Trong deep learning, các hàm mất mát thường được sử dụng để đo lường mức độ sai lệch giữa dự đoán của mô hình và giá trị thực tế. Mục tiêu của quá trình huấn luyện mô hình là tối thiểu hóa hàm mất mát này. Một số hàm mất mát phổ biến trong deep learning, chẳng hạn như hàm mất mát bình phương trung bình (mean squared error), hàm mất mát cross-entropy, thường được thiết kế để là các hàm lồi.

Hàm `exp` (hàm mũ) và `log` (logarithm) không phải là hàm lồi trên toàn miền của chúng. Tuy nhiên, chúng có thể là hàm lồi trên một khoảng cụ thể của miền của chúng. Ví dụ, hàm `exp` là lồi trên toàn bộ miền số thực, trong khi hàm `log` là lồi trên miền dương của nó.

Khi được sử dụng trong deep learning, hàm mũ và logarithm thường xuất hiện trong các biểu thức tính toán gradient (đạo hàm) của hàm mất mát. Các phép toán này có thể giúp đơn giản hóa việc tính toán đạo hàm và tối ưu hóa mô hình.


Một số ví dụ và chứng minh một hàm là hàm lồi
======

![img](/images/convex_function/1.JPG)


![img](/images/convex_function/2.JPG){: .align-center width="500px"}


![img](/images/convex_function/3.JPG){: .align-center width="500px"}


![img](/images/convex_function/4.JPG){: .align-center width="500px"}


![img](/images/convex_function/5.JPG){: .align-center width="500px"}


![img](/images/convex_function/6.JPG){: .align-center width="500px"}


Hết.
