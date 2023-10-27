---
title: '[Queuing Theory and Algorithms] Một số khái niệm cơ bản'
date: 2023-10-20
permalink: /posts/2023/10/20/queueing-theory-and-algorithm-mot-so-khai-niem-co-ban/
tags:
  - queuing theory
  - algorithms
---

Một số khái niệm cơ bản của lý thuyết hàng đợi và giải thích các khái niệm liên quan

Lý thuyết hàng đợi
======

Lý thuyết Hàng đợi (Queueing Theory) là một lĩnh vực trong khoa học máy tính, toán học, kỹ thuật và các lĩnh vực liên quan đến quản lý dịch vụ. Nó nghiên cứu về các hệ thống có sự điều phối của các đối tượng 
(ví dụ: khách hàng, gói tin dữ liệu) đến và rời đi từ một điểm xử lý.

Dưới đây là một số khái niệm cơ bản trong Lý thuyết hàng đợi:

* Hàng đợi (Queue): Đây là nơi các đối tượng đến và chờ đợi để được phục vụ. Ví dụ: trong mạng, các gói tin có thể tạo thành hàng đợi tại các nút truyền thông.

* Đối tượng (Entity): Đây là cái gì đó cần được xử lý. Ví dụ: trong một hệ thống hàng đợi đơn giản, đối tượng có thể là người chờ đợi để được phục vụ.

* Kênh (Server): Kênh là nơi xử lý các đối tượng. Trong một hệ thống, có thể có một hoặc nhiều kênh. Ví dụ: trong một trạm truyền thông, có thể có nhiều kênh xử lý các gói tin.

* Tốc độ phục vụ (Service Rate): Đây là tốc độ mà các đối tượng được xử lý hoặc phục vụ. Nó có thể được biểu thị bằng số đối tượng xử lý trong một đơn vị thời gian.

* Tốc độ đến (Arrival Rate): Đây là tốc độ mà các đối tượng đến hàng đợi. Nó có thể được biểu thị bằng số đối tượng đến trong một đơn vị thời gian.

* Độ trễ (Delay): Đây là thời gian mà một đối tượng phải đợi trong hàng đợi trước khi được phục vụ.

* Tỷ lệ phục vụ (Utilization): Tỷ lệ phần trăm thời gian kênh hoặc thiết bị xử lý thực sự đang được sử dụng để phục vụ.

* Hệ số dịch vụ (Service Discipline): Đây là cách thức xác định đối tượng nào sẽ được phục vụ tiếp theo khi có sự cạnh tranh.

* Cấu trúc hệ thống hàng đợi (Queueing System Structure): Gồm các yếu tố như số lượng kênh, cách thức đến, cách thức phục vụ, v.v.


Độ trễ trung bình mạng
======

Độ trễ mạng trung bình là thời gian trung bình mà dữ liệu mất để di chuyển từ nguồn đến đích qua mạng. Đây là một yếu tố quan trọng trong đánh giá hiệu suất mạng. 
Độ trễ mạng bao gồm nhiều thành phần như độ trễ chuyển tiếp (propagation delay), độ trễ xử lý (processing delay), độ trễ hàng đợi (queuing delay), và độ trễ truyền tải (transmission delay).

### 1. Độ trễ chuyển tiếp (Propagation Delay):

* Độ trễ này là thời gian mà tín hiệu mất để đi từ nguồn đến đích thông qua truyền tải vật lý (như cáp, sợi quang, không khí).
* Phụ thuộc vào khoảng cách vật lý giữa các thiết bị và tốc độ truyền tải của môi trường truyền tải.

### 2. Độ trễ xử lý (Processing Delay):

* Độ trễ này liên quan đến thời gian mà các thiết bị mạng (như router hoặc switch) mất để kiểm tra và xử lý gói tin.
* Bao gồm thời gian kiểm tra lỗi, quyết định định tuyến, và các hoạt động liên quan khác.

### 3. Độ trễ hàng đợi (Queuing Delay):

* Khi mạng bị quá tải, gói tin có thể phải đợi trong hàng đợi tại các thiết bị mạng (router) trước khi được xử lý.
* Độ trễ này tăng lên khi có nhiều gói tin cần xử lý hơn so với khả năng xử lý của thiết bị.

### 4. Độ trễ truyền tải (Transmission Delay):

* Độ trễ này là thời gian mà gói tin mất để truyền từ nguồn đến đích trên một đường truyền vật lý.
* Phụ thuộc vào kích thước gói tin và băng thông của đường truyền.


Khi so sánh mạng bình thường và mạng vệ tinh, có một số sự khác biệt quan trọng về độ trễ:

* Độ trễ truyền thống (Propagation Delay): Đây là thời gian mất đi khi dữ liệu phải di chuyển qua không gian. Trong mạng vệ tinh, độ trễ truyền thống có thể cao hơn do tần số cao và khoảng cách xa.
* Độ trễ xử lý (Processing Delay): Đây là thời gian mất đi do việc xử lý tại các nút mạng (như router, switch). Mạng vệ tinh thường có độ trễ xử lý cao hơn do việc phải đi qua các trạm đất (ground stations) trước khi dữ liệu được chuyển đến không gian.
* Độ trễ lưu trữ (Storage Delay): Đây là thời gian mất đi do việc lưu trữ tạm thời dữ liệu tại các điểm trung gian trong mạng. Mạng vệ tinh cũng có thể gặp vấn đề này do cần phải lưu trữ tạm thời dữ liệu tại các trạm đất.
* Độ trễ kiểm soát (Control Delay): Đây liên quan đến thời gian mất đi do việc quản lý các thông tin điều khiển trên mạng. Mạng vệ tinh có thể gặp thách thức này do các tín hiệu điều khiển phải đi qua không gian.


Các yếu tố quan trọng trong lý thuyết hàng đợi
======

### 1. Đơn vị phục vụ (Service Unit): 

* Đây là công việc cần thực hiện bởi hệ thống hàng đợi, như việc xử lý một yêu cầu mạng hoặc phục vụ một khách hàng.

### 2. Độ tải (Traffic Intensity): 

* Được đo bằng tỷ lệ giữa tốc độ đến của công việc (yêu cầu mới) và tốc độ phục vụ (sự hoàn thành của công việc). Khi độ tải lớn hơn 1, hàng đợi sẽ tăng lên vô hạn.

### 3. Loại hàng đợi (Queue Discipline): 

* Xác định cách thức chọn công việc từ hàng đợi khi có nhiều công việc đang chờ đợi.

### 4. Số kênh phục vụ (Number of Service Channels): 

* Số lượng kênh có sẵn để phục vụ công việc. Mỗi kênh có thể xử lý một công việc tại một thời điểm.

### 5. Thời gian phục vụ (Service Time): 

* Thời gian mà mỗi công việc mất để hoàn thành khi đang được phục vụ.


Loại hàng đợi (Queue Discipline) 
======

### First-Come-First-Serve (FCFS) 

Đây là một trong những phương pháp đơn giản nhất và phổ biến nhất trong lý thuyết hàng đợi. Nó dựa trên nguyên tắc đơn giản: "Gói tin đến sớm nhất được xử lý đầu tiên". Dưới đây là một trình bày chi tiết về FCFS:

#### 1. **Nguyên tắc hoạt động:**
   
   * Khi một gói tin đến tại máy chủ hoặc router, nó sẽ được thêm vào cuối hàng đợi.
   * Gói tin đầu tiên được đẩy ra khỏi hàng đợi để được xử lý.
   * Sau khi gói tin đầu tiên đã được xử lý hoàn toàn, gói tin tiếp theo trong hàng đợi sẽ được xử lý.

#### 2. **Ưu điểm:**
   
   * FCFS rất dễ triển khai và không đòi hỏi nhiều tính toán phức tạp.
   * Đối với các gói tin cùng đến cùng một thời điểm, họ được xử lý theo thứ tự mà chúng đến.

#### 3. **Nhược điểm:**
   
   * Không quan tâm đến ưu tiên:** FCFS không xem xét đến ưu tiên của các gói tin. Các gói tin quan trọng có thể phải chờ lâu nếu có nhiều gói tin khác đang chờ đợi.

#### 4. **Ví dụ cụ thể:**
   
   * Giả sử có một hàng đợi gồm 3 gói tin đến lần lượt vào các thời điểm 10, 20 và 15. FCFS sẽ xử lý các gói tin theo thứ tự như sau: Gói tin 1 (đến lúc 10), sau đó Gói tin 3 (đến lúc 15), cuối cùng là Gói tin 2 (đến lúc 20).

#### 5. **Ứng dụng:**
   
   * FCFS thường được sử dụng trong các tình huống đòi hỏi tính chất công bằng và không có yêu cầu ưu tiên đặc biệt.


### Weighted Fair Queueing (WFQ) 

Đây là một phương pháp quản lý hàng đợi mạng được sử dụng để phân phối tài nguyên mạng dựa trên trọng số được gán cho từng loại lưu lượng. Phương pháp này nhằm tạo ra một sự cân bằng tốt giữa các loại lưu lượng khác nhau. Dưới đây là một trình bày chi tiết về WFQ:

#### 1. **Nguyên tắc hoạt động:**
   
   * Mỗi loại lưu lượng được gán một trọng số (weight). Trọng số này phản ánh mức độ ưu tiên của loại lưu lượng đó so với các loại khác.
   * Khi một gói tin đến, hệ thống xác định loại lưu lượng của nó và dựa trên trọng số, xác định thời gian dành cho loại lưu lượng đó để sử dụng tài nguyên mạng.

#### 2. **Tính chất chính:**

   * Cân bằng trọng số: WFQ đảm bảo rằng các loại lưu lượng được phục vụ với tần suất tương ứng với trọng số của chúng.
   * Cung cấp sự linh hoạt: Cho phép ưu tiên cao hơn cho các loại lưu lượng được gán trọng số cao.

#### 3. **Cách thức hoạt động cụ thể:**
   
   * Khi gói tin đến, hệ thống xác định loại lưu lượng và gán thời gian dự kiến để sử dụng tài nguyên mạng dựa trên trọng số của loại lưu lượng đó.
   * Các loại lưu lượng được phục vụ theo thứ tự tăng dần của thời gian được gán cho mỗi loại.
   * Nếu các loại lưu lượng có cùng thời gian được gán, gói tin sẽ được phục vụ theo nguyên tắc FCFS.

#### 4. **Ưu điểm:**
   
   * Cân bằng và công bằng: Đảm bảo rằng các loại lưu lượng được phục vụ với tần suất tương ứng với trọng số của chúng.
   * Cho phép ưu tiên hóa: Có thể gán trọng số cao hơn cho các loại lưu lượng quan trọng hơn.

#### 5. **Nhược điểm:**
   
   * Đòi hỏi tính toán phức tạp: WFQ đòi hỏi tính toán để xác định thời gian dành cho mỗi loại lưu lượng.

#### 6. **Ứng dụng:**
   
   * WFQ thường được sử dụng trong các môi trường đòi hỏi sự cân bằng giữa các loại lưu lượng khác nhau và có yêu cầu đặc biệt về ưu tiên. Đặc biệt, nó thường được sử dụng trong các mạng hoạt động thương mại hoặc trong các hệ thống nơi cần quản lý nhiều loại lưu lượng cùng một lúc.


### Priority Queueing (PQ) 

Đây là một phương pháp quản lý hàng đợi mạng mà các gói tin được ưu tiên xử lý dựa trên mức độ ưu tiên được gán cho chúng. Dưới đây là một trình bày chi tiết về Priority Queueing (PQ):

#### 1. **Nguyên tắc hoạt động:**
   
   * Mỗi gói tin được gán một mức độ ưu tiên (priority level) dựa trên các tiêu chí như loại lưu lượng, nguồn gốc, hoặc yêu cầu cụ thể của mạng.
   * Các gói tin với mức độ ưu tiên cao hơn được ưu tiên và được xử lý trước các gói tin với mức độ ưu tiên thấp hơn.

#### 2. **Tính chất chính:**
   * Độ ưu tiên cao được phục vụ trước: PQ đảm bảo rằng các gói tin với mức độ ưu tiên cao hơn sẽ được xử lý trước các gói tin với mức độ ưu tiên thấp hơn.

#### 3. **Cách thức hoạt động cụ thể:**
   
   * Khi gói tin đến, hệ thống xác định mức độ ưu tiên của nó và xếp vào hàng đợi tương ứng với mức độ ưu tiên đó.
   * Các gói tin được xử lý theo thứ tự ưu tiên, với các gói tin có mức độ ưu tiên cao hơn được xử lý trước.

#### 4. **Ưu điểm:**
   
   * Ưu tiên ứng dụng dễ dàng: PQ cho phép đảm bảo ưu tiên cho các gói tin quan trọng hoặc yêu cầu đặc biệt.
   * Được sử dụng trong các tình huống đòi hỏi ưu tiên cao hơn: PQ thường được sử dụng trong các môi trường mạng nơi các yêu cầu về ưu tiên cao hơn là quan trọng.

#### 5. **Nhược điểm:**
   
   * Không quản lý tình trạng quá tải: PQ không quản lý tình trạng quá tải của mạng. Điều này có thể dẫn đến tình trạng quá tải nếu có quá nhiều gói tin ưu tiên.

#### 6. **Ứng dụng:**
   
   * PQ thường được sử dụng trong các môi trường mạng nơi các yêu cầu về ưu tiên cao hơn là quan trọng, ví dụ như trong các mạng y tế, tài chính hoặc các hệ thống đòi hỏi độ tin cậy cao.


### Random Early Detection (RED) 

Đây là một phương pháp quản lý hàng đợi mạng được sử dụng để tránh tình trạng quá tải bằng cách loại bỏ gói tin một cách ngẫu nhiên khi bộ đệm đạt một mức độ quá tải nhất định. Dưới đây là một trình bày chi tiết về RED:

#### 1. **Nguyên tắc hoạt động:**
   
   * Khi một gói tin đến, hệ thống kiểm tra mức độ quá tải của bộ đệm (buffer occupancy).
   * Nếu mức độ quá tải chưa đạt ngưỡng được thiết lập trước (min-threshold), gói tin được chấp nhận và đưa vào hàng đợi.
   * Nếu mức độ quá tải vượt ngưỡng min-threshold nhưng chưa đạt ngưỡng max-threshold, gói tin có một xác suất nhất định bị loại bỏ.
   * Nếu mức độ quá tải vượt ngưỡng max-threshold, gói tin sẽ bị loại bỏ mà không cần xem xét xác suất.

#### 2. **Tính chất chính:**
   
   * Quản lý tình trạng quá tải: RED giúp kiểm soát tình trạng quá tải bằng cách loại bỏ gói tin một cách ngẫu nhiên khi bộ đệm sắp quá tải.
   * Cung cấp cơ hội cho các gói tin nhạy cảm đến thời gian phục vụ đủ lớn: Các gói tin nhạy cảm với thời gian phục vụ lớn hơn có cơ hội cao hơn để được xử lý.

#### 3. **Cách thức hoạt động cụ thể:**
   
   * Khi gói tin đến, hệ thống xác định mức độ quá tải của bộ đệm dựa trên ngưỡng min-threshold và max-threshold được cấu hình trước.
   * Dựa trên mức độ quá tải, gói tin có thể được chấp nhận, hoặc có một xác suất nhất định bị loại bỏ, hoặc bị loại bỏ ngay lập tức.

#### 4. **Ưu điểm:**
   
   * Phòng ngừa tình trạng quá tải: RED giúp tránh tình trạng quá tải trong hệ thống mạng bằng cách loại bỏ gói tin một cách ngẫu nhiên khi bộ đệm đạt mức quá tải cao.
   * Cung cấp sự linh hoạt trong quản lý bộ đệm: Các ngưỡng min-threshold và max-threshold có thể được điều chỉnh để điều chỉnh hoạt động của RED.

##### 5. **Nhược điểm:**
   
   * Đòi hỏi cấu hình chính xác: RED đòi hỏi việc cấu hình kỹ lưỡng để đảm bảo hoạt động hiệu quả.

#### 6. **Ứng dụng:**
   
   * RED thường được sử dụng trong các mạng nơi tình trạng quá tải có thể xảy ra và cần phải ngăn chặn để tránh mất mát dữ liệu.


### Round Robin (RR) 

Đây là một phương pháp quản lý hàng đợi mạng mà các gói tin từ các loại lưu lượng khác nhau được xử lý theo một trình tự vòng lặp. Dưới đây là một trình bày chi tiết về Round Robin (RR):

#### 1. **Nguyên tắc hoạt động:**
   
   * Các loại lưu lượng được sắp xếp thành một danh sách và xử lý theo trình tự tuần tự, từ loại đầu tiên đến loại cuối cùng.
   * Mỗi loại lưu lượng được phục vụ một lượng nhỏ (được gọi là time quantum hoặc slice) trước khi chuyển sang loại lưu lượng tiếp theo.

#### 2. **Tính chất chính:**
   
   * Công bằng: RR đảm bảo rằng mỗi loại lưu lượng đều được phục vụ và không bị ưu tiên quá mức so với các loại khác.
   * Phản ứng nhanh với các loại lưu lượng đến đột ngột: Do mỗi loại lưu lượng chỉ được phục vụ trong một khoảng thời gian ngắn, RR có thể phản ứng nhanh với các loại lưu lượng đến đột ngột.

#### 3. **Cách thức hoạt động cụ thể:**
   
   * Khi gói tin đến, hệ thống xác định loại lưu lượng của nó và đưa vào danh sách xếp hàng của loại lưu lượng đó.
   * Các loại lưu lượng được phục vụ theo thứ tự từng lượt (round), với mỗi loại lưu lượng được phục vụ trong một khoảng thời gian ngắn (time quantum).
   * Nếu loại lưu lượng có nhiều gói tin hơn trong danh sách xếp hàng, các gói tin tiếp theo của loại đó sẽ được đưa vào danh sách xếp hàng của loại lưu lượng tiếp theo.

#### 4. **Ưu điểm:**
   
   * Công bằng và đáng tin cậy: RR đảm bảo mỗi loại lưu lượng đều được phục vụ, không bị ưu tiên quá mức.
   * Phản ứng nhanh với các loại lưu lượng đến đột ngột: Do mỗi loại lưu lượng chỉ được phục vụ trong một khoảng thời gian ngắn, RR có thể phản ứng nhanh với các loại lưu lượng đến đột ngột.

#### 5. **Nhược điểm:**
   
   * Không quản lý tình trạng quá tải: RR không cung cấp cơ chế để quản lý tình trạng quá tải của mạng.

#### 6. **Ứng dụng:**
   
   * RR thường được sử dụng trong các tình huống cần đảm bảo công bằng và không ưu tiên đặc biệt, và không có yêu cầu đặc biệt về quản lý tình trạng quá tải mạng.


Queueing theory algorithms
======

### 1. **M/M/1 Queue**:
   
   * M (Markovian Arrival Process): Thời gian giữa các sự kiện đến hàng đợi là biến ngẫu nhiên tuân theo quá trình Markov, thường được mô tả bằng phân phối Poisson.
   * M (Exponential Service Time): Thời gian phục vụ của mỗi khách hàng tuân theo phân phối mũ (Exponential).
   * 1 (Single Server): Hệ thống chỉ có một server duy nhất để phục vụ khách hàng.

### 2. **M/M/c Queue**:
   
   * Tương tự như M/M/1 nhưng với nhiều server (c). Điều này cho phép nhiều khách hàng cùng được phục vụ đồng thời.

### 3. **M/G/1 Queue**:

   * M (Markovian Arrival Process): Như trên.
   * G (General Service Time): Thời gian phục vụ của mỗi khách hàng không còn tuân theo phân phối mũ, mà có thể tuân theo bất kỳ phân phối nào.
   * 1 (Single Server): Hệ thống chỉ có một server.

### 4. **M/G/c Queue**:
   
   * Tương tự như M/G/1 nhưng với nhiều server (c).

### 5. **M/D/1 Queue**:
   
   * D(Deterministic Service Time): Thời gian phục vụ là một giá trị cố định, không ngẫu nhiên.
   * 1(Single Server): Hệ thống chỉ có một server.

### 6. **M/D/c Queue**:
   
   * Tương tự như M/D/1 nhưng với nhiều server (c).

### 7. **Hàng đợi M/M/1/K**: 

   * Markovian Arrival Process): Thời gian giữa các sự kiện đến hàng đợi tuân theo quá trình Markov, thường được mô tả bằng phân phối Poisson.
   * Exponential Service Time: Thời gian phục vụ của mỗi khách hàng tuân theo phân phối mũ Exponential.
   * Single Server: Hệ thống chỉ có một server duy nhất để phục vụ khách hàng.
   * K: Số lượng tối đa khách hàng mà hàng đợi có thể chứa. Khi hàng đợi đạt đến sức chứa tối đa, các khách hàng tiếp theo sẽ bị từ chối hoặc rơi vào trạng thái blocked chờ đợi.

### 8. So sánh M/M/1 vs M/M/1/K

   * Nếu dùng M/M/1 thì khó chỉ ra được sự khác biệt khi đề cập đến packet loss, vì khi gói tin đến. Ví dụ M/M/1/k (k = 3) thì có nghĩa là 3 gói tin sẽ được server chờ đợi để đáp ứng.
   * Còn nếu gói thứ 4 đến thì nó sẽ bị drop dẫn đến packet loss. Còn đối với M/M/1 thì queue là vô hạn nên không thể đo được packet losss.

Tài liệu tham khảo: [Queueing theory and algorithms](https://packetpushers.net/average-network-delay/)

Hết.
