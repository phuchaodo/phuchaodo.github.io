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


Ví dụ về hàm lồi
======

Để hiểu rõ hơn về hàm lồi, hãy xem xét một số ví dụ cụ thể.

1. **Hàm tuyến tính**:
   - Ví dụ: \(f(x) = ax + b\), với \(a\) và \(b\) là các hằng số.
   - Đây là một hàm tuyến tính và cũng là một ví dụ của hàm lồi. Bất kỳ hai điểm nào trên đồ thị của hàm này, đoạn thẳng nối chúng sẽ luôn nằm phía trên đồ thị.

2. **Hàm mũ**:
   - Ví dụ: \(f(x) = e^x\).
   - Hàm mũ cũng là một ví dụ của hàm lồi. Trên toàn miền của nó, đồ thị sẽ có hình dạng uốn lượn nhưng không bao giờ có phần nào nằm phía dưới đồ thị.

3. **Hàm lượng tâm**:
   - Ví dụ: \(f(x) = x^2\).
   - Đây là một hàm bậc hai, và cũng là một ví dụ của hàm lồi. Bạn có thể thấy rằng bất kỳ đoạn thẳng nào nối hai điểm bất kỳ trên đồ thị của nó đều nằm phía dưới đồ thị.

4. **Hàm giảm dần (decreasing function)**:
   - Ví dụ: \(f(x) = -x\).
   - Hàm này cũng là lồi. Bất kỳ đoạn thẳng nào nối hai điểm trên đồ thị của nó đều nằm phía dưới đồ thị.

5. **Hàm không lồi**:
   - Ví dụ: \(f(x) = x^3\).
   - Hàm này không phải là lồi. Bạn có thể tìm thấy các điểm trên đồ thị mà đoạn thẳng nối chúng nằm phía dưới đồ thị.


Chứng minh một hàm là hàm lồi
======

Để chứng minh rằng một hàm \(f(x)\) là hàm lồi trên một miền cụ thể, ta cần kiểm tra điều kiện sau:

1. **Định nghĩa hàm lồi**:

   Một hàm \(f(x)\) được gọi là lồi trên một miền \(D\) nếu với mọi \(x_1, x_2\) trong \(D\) và mọi số thực \(t\) từ 0 đến 1, ta có:

   \[f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2)\]

2. **Kiểm tra từng bước**:

   - **Bước 1**: Chọn hai điểm bất kỳ \(x_1\) và \(x_2\) trong miền \(D\) của hàm.
   
   - **Bước 2**: Chọn một số thực \(t\) từ khoảng từ 0 đến 1.
   
   - **Bước 3**: Tính \(tx_1 + (1-t)x_2\).
   
   - **Bước 4**: Tính \(tf(x_1) + (1-t)f(x_2)\).
   
   - **Bước 5**: Tính \(f(tx_1 + (1-t)x_2)\).
   
   - **Bước 6**: So sánh kết quả từ bước 5 với kết quả từ bước 4. Nếu điều kiện \(f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2)\) được thỏa mãn, thì hàm \(f(x)\) là lồi trên miền \(D\).

3. **Chứng minh tiếp với các trường hợp cụ thể**:

   - Nếu ta có các hàm phức tạp, cần kiểm tra điều kiện lồi trên từng miền cụ thể.

Lưu ý rằng không phải tất cả các hàm đều là lồi, và cách chứng minh lồi sẽ khác nhau tùy thuộc vào tính chất của hàm đó. Trong thực tế, việc chứng minh tính lồi của một hàm có thể khá phức tạp và đòi hỏi kiến thức về phân tích toán học và đại số tuyến tính.


Một ví dụ
======

Tất nhiên, hãy xem xét ví dụ các hàm và chứng minh tính lồi của chúng.

1. **Hàm logarithm tự nhiên (\(f(x) = \log(x)\)) trên miền dương**:

   - Đầu tiên, chúng ta chọn hai điểm bất kỳ \(x_1\) và \(x_2\) trong miền dương (\(x_1 > 0\) và \(x_2 > 0\)).
   
   - Chọn một số thực \(t\) từ khoảng từ 0 đến 1.
   
   - Tính \(tx_1 + (1-t)x_2\).
   
   - Tính \(tf(x_1) + (1-t)f(x_2)\).
   
   - So sánh \(f(tx_1 + (1-t)x_2)\) với \(tf(x_1) + (1-t)f(x_2)\).
   
   - Chúng ta muốn chứng minh rằng:
     \[\log(tx_1 + (1-t)x_2) \leq t\log(x_1) + (1-t)\log(x_2)\]
   
   - Chứng minh: Với \(t\) và \(x_1, x_2\) được chọn như trên, ta có:
     \[tx_1 + (1-t)x_2 > 0\]
     Do đó, hàm logarithm tự nhiên là lồi trên miền dương của nó.

2. **Hàm mũ (\(f(x) = e^x\)) trên toàn miền số thực**:

   - Chúng ta chọn hai điểm bất kỳ \(x_1\) và \(x_2\) trong miền số thực (\(x_1, x_2 \in \mathbb{R}\)).
   
   - Chọn một số thực \(t\) từ khoảng từ 0 đến 1.
   
   - Tính \(tx_1 + (1-t)x_2\).
   
   - Tính \(tf(x_1) + (1-t)f(x_2)\).
   
   - So sánh \(f(tx_1 + (1-t)x_2)\) với \(tf(x_1) + (1-t)f(x_2)\).
   
   - Chúng ta muốn chứng minh rằng:
     \[e^{tx_1 + (1-t)x_2} \leq te^{x_1} + (1-t)e^{x_2}\]
   
   - Chứng minh: Sử dụng tính chất của hàm mũ, ta có:
     \[e^{tx_1 + (1-t)x_2} = e^{tx_1} \cdot e^{(1-t)x_2} \leq te^{x_1} + (1-t)e^{x_2}\]
     Do đó, hàm mũ là lồi trên toàn miền số thực.

3. **Hàm tuyến tính (\(f(x) = ax + b\)) trên toàn miền số thực**:

   - Chúng ta chọn hai điểm bất kỳ \(x_1\) và \(x_2\) trong miền số thực (\(x_1, x_2 \in \mathbb{R}\)).
   
   - Chọn một số thực \(t\) từ khoảng từ 0 đến 1.
   
   - Tính \(tx_1 + (1-t)x_2\).
   
   - Tính \(tf(x_1) + (1-t)f(x_2)\).
   
   - So sánh \(f(tx_1 + (1-t)x_2)\) với \(tf(x_1) + (1-t)f(x_2)\).
   
   - Chúng ta muốn chứng minh rằng:
     \[a(tx_1 + (1-t)x_2) + b \leq t(ax_1 + b) + (1-t)(ax_2 + b)\]
   
   - Chứng minh: Thực hiện các phép toán và thu gọn, ta có:
     \[a(tx_1 + (1-t)x_2) + b = t(ax_1 + b) + (1-t)(ax_2 + b)\]
     Do đó, hàm tuyến tính là lồi trên toàn miền số thực.


Hết.
