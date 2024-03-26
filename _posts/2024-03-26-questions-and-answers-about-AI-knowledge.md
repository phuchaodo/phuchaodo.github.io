---
title: '[Note] Questions and answers about AI knowledge'
date: 2024-03-26
permalink: /posts/2024/03/26/questions-and-answers-about-AI-knowledge/
tags:
  - research
  - proposal
  - AI
  - Knowledge
--- 

Questions and answers about AI knowledge

## List question

1. Khi loss function nó giảm nhưng lâu lâu lại tăng đột ngột  thì xảy ra hiên tượng gì. Cách khắc phục.

2. hàm ReLU dùng để làm gì? Hàm sigmoid để làm gì? Sự khác nhau của nó?

3. Matrix giả nghịch đảo để làm gì

4. Relu có ưu điểm gì,thay bằng sigmoid có dc ko

5. Nhược điểm lớn nhất của linear regession

6. Làm sao tìm dc global mininum khi có nhiều local minimum.

7. Kiểm định chi-square dùng để làm gì, nguồn gốc của phân phối chi-square

8. P value là gì, giá trị bao nhiêu là tốt(cho con số cụ thể)

9. Trình bày ý nghĩa của Batch Normalization

10. Trình bày khái niệm và mối quan hệ đánh đổi giữa bias và variance?

11. Giả sử sau mô hình Deep Learning tìm được 10 triệu vector khuôn mặt. Làm sao tìm query khuôn mặt mới nhanh nhất.

12. Với bài toán classification, chỉ số accuracy có hoàn toàn tin tưởng được không. Bạn thường sử dụng các độ đo nào để đánh giá mô hình của mình ?

13. Bạn hiểu như thế nào về Backpropagation? Giải thích cơ chế hoạt động?

14. Ý nghĩa của hàm activation function là gì? Thế nào là điểm bão hòa của các activation functions?

15. Hyperparameters của mô hình là gì? Khác với parameters như thế nào?

16. Điều gì xảy ra khi learning rate quá lớn hoặc quá nhỏ?

17. Khi kích thước ảnh đầu vào tăng gấp đôi thì số lượng tham số của CNN tăng lên bao nhiêu lần? Tại sao?

18. Với những tập dữ liệu bị imbalance thì có những cách xử lý nào?

19. Các khái niệm Epoch, Batch và Iteration có ý nghĩa gì khi training mô hình Deep Learning

20. Khái niệm Data Generator là gì? Cần dùng nó khi nào?

21. Phân biệt scalars, vectors, ma trận, và tensors?

22. Chuẩn norm của vecto và ma trận là gì?

23. Đạo hàm là gì?

24. Trị riêng và vecto riêng là gì? Nêu một vài tính chất của chúng.

25. Xác suất là gì? Tại sao nên sử dụng xác suất trong machine learning?

26. Biến ngẫu nhiên là gì? Nó khác gì so với biến đại số thông thường?

27. Xác suất có điều kiện là gì? Cho ví dụ?

28. Khái niệm về kỳ vọng, phương sai và ý nghĩa của chúng?

## Question from chatgpt

**For AI Engineers:**

1. Explain the difference between supervised and unsupervised learning. Provide examples of each.

2. What are some common activation functions used in neural networks, and when would you use each?

3. How do you handle overfitting in machine learning models?

4. Describe the backpropagation algorithm and its role in training neural networks.

5. What is reinforcement learning, and how does it differ from supervised and unsupervised learning?

6. Can you explain the concept of regularization in machine learning? How does it work, and why is it important?

7. How do you evaluate the performance of a machine learning model?

8. What is cross-validation, and why is it useful?

9. Explain the concept of feature engineering and its importance in machine learning.

10. What is the curse of dimensionality, and how does it affect machine learning algorithms?

11. Can you discuss the differences between traditional machine learning algorithms and deep learning algorithms?

12. How would you approach a problem where you have a large amount of unlabeled data?

13. Describe the bias-variance tradeoff and its implications for machine learning models.

14. What are some common techniques for reducing dimensionality in machine learning?

15. How would you deploy a machine learning model into production?

**For Data Scientists:**

1. What is the difference between supervised and unsupervised learning?

2. Can you explain the steps you would take to clean and preprocess a dataset?

3. How do you handle missing data in a dataset?

4. What is the purpose of exploratory data analysis (EDA), and what techniques do you use for EDA?

5. Explain the concept of feature selection and feature importance.

6. What are some common machine learning algorithms you have used, and in what situations would you use each?

7. How do you deal with imbalanced datasets?

8. Can you discuss the differences between classification and regression algorithms?

9. How would you assess the performance of a classification model?

10. What is cross-validation, and why is it important?

11. Describe the difference between correlation and causation. Why is it important to understand this difference in data analysis?

12. What is regularization, and why is it used in machine learning models?

13. Can you explain the concept of clustering and give an example of when it might be used?

14. How would you communicate the results of your analysis to stakeholders who may not have a technical background?

15. What tools and programming languages are you proficient in for data analysis and visualization?


**For AI Engineers:**

16. What are convolutional neural networks (CNNs), and what are they commonly used for?

17. Explain the concept of transfer learning in deep learning. How does it work, and when is it beneficial?

18. What are recurrent neural networks (RNNs), and what are some applications where they excel?

19. Can you discuss the differences between batch gradient descent, stochastic gradient descent, and mini-batch gradient descent?

20. What is the vanishing gradient problem, and how can it be mitigated?

21. Describe the concept of attention mechanisms in deep learning. How are they used, and what advantages do they offer?

22. How do you choose the appropriate neural network architecture for a given problem?

23. What are generative adversarial networks (GANs), and how do they work?

24. Explain the concept of word embeddings. What are they used for, and how are they trained?

25. Can you discuss some common techniques for optimizing neural network performance, such as dropout, batch normalization, and learning rate scheduling?

26. What is the difference between a feedforward neural network and a recurrent neural network?

27. How do you handle non-numerical data (e.g., text or images) in a machine learning model?

28. Discuss some challenges associated with training deep neural networks.

29. Can you explain the concept of transfer learning in the context of natural language processing (NLP)?

30. How would you approach a problem involving time series forecasting using deep learning techniques?

**For Data Scientists:**

16. What is the difference between correlation and causation, and why is it important in data analysis?

17. Can you discuss the concept of feature scaling and its importance in machine learning?

18. What are some common methods for dealing with outliers in a dataset?

19. How would you handle categorical variables in a machine learning model?

20. Explain the concept of bias in machine learning models. How can bias be identified and addressed?

21. What is the purpose of regularization in linear regression, and what are some common regularization techniques?

22. Can you describe the process of hyperparameter tuning and its significance in machine learning?

23. How would you approach a problem where the data is too large to fit into memory?

24. What are some techniques for handling multicollinearity in regression analysis?

25. Discuss the advantages and disadvantages of different types of machine learning models, such as decision trees, support vector machines, and neural networks.

26. How do you handle time-series data in machine learning models?

27. What is the difference between overfitting and underfitting, and how do you prevent them in machine learning models?

28. Can you explain the concept of ensemble learning and give examples of ensemble methods?

29. How do you interpret the coefficients in a logistic regression model?

30. What are some common evaluation metrics used for regression tasks, and how do you interpret them?


**For AI Engineers:**

31. Explain the concept of hyperparameter tuning and its importance in training machine learning models.

32. What is the role of optimization algorithms in training neural networks? Discuss some common optimization algorithms and their characteristics.

33. Can you describe the architecture of a typical convolutional neural network (CNN) used for image classification?

34. Discuss the concept of recurrent neural networks (RNNs) and their applications in sequential data processing.

35. How do you handle class imbalances in classification tasks, especially in scenarios where one class significantly outnumbers the others?

36. Can you explain the concept of attention mechanisms in neural networks, particularly in the context of natural language processing (NLP)?

37. What are autoencoders, and how are they used for dimensionality reduction and anomaly detection?

38. Describe the concept of generative models and their applications in generating realistic data, such as images or text.

39. How do you preprocess text data for natural language processing tasks, including tokenization, stemming, and lemmatization?

40. Discuss the challenges and techniques involved in training deep learning models on limited computational resources, such as edge devices or mobile phones.

41. Can you explain the concept of adversarial attacks in deep learning, and how can models be made more robust against such attacks?

42. How would you handle noisy data in a machine learning pipeline, particularly in scenarios where the noise might be detrimental to model performance?

43. Discuss the concept of transfer learning in computer vision tasks, including fine-tuning pretrained models for specific domains or tasks.

44. What are some strategies for deploying machine learning models at scale, considering factors such as scalability, latency, and reliability?

45. Can you discuss the ethical considerations and potential biases associated with deploying AI systems in real-world applications, and how would you address them?

**For Data Scientists:**

31. How do you assess the importance of features in a machine learning model, and what techniques can be used for feature selection?

32. Discuss the differences between batch processing and streaming processing in the context of big data analytics.

33. Can you explain the concept of A/B testing and how it is used to evaluate the effectiveness of changes or interventions?

34. How would you handle skewed distributions in a dataset, particularly when building predictive models?

35. What are some common methods for imputing missing values in a dataset, and how do you decide which method to use?

36. Discuss the process of feature extraction from unstructured data sources such as text or images.

37. How do you assess the multicollinearity of features in a regression model, and what are the potential consequences of multicollinearity?

38. Can you explain the concept of anomaly detection and some techniques used to identify outliers in a dataset?

39. Discuss the trade-offs between model interpretability and model complexity in machine learning, and how would you decide which approach to prioritize in a given scenario?

40. How do you handle time-series data with seasonality and trends when building forecasting models?

41. What are some common techniques for reducing dimensionality in high-dimensional datasets, and how do you evaluate the impact of dimensionality reduction on model performance?

42. Can you describe the process of natural language processing (NLP) pipeline, including tokenization, part-of-speech tagging, and named entity recognition?

43. How do you validate the results of a machine learning model, particularly in scenarios where ground truth labels may be unavailable or difficult to obtain?

44. Discuss the differences between supervised, unsupervised, and semi-supervised learning, and provide examples of each.

45. Can you discuss some recent advancements or trends in the field of data science, and how do you stay updated with the latest developments in the field?


Hết.
