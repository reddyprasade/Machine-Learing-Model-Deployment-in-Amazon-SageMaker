# Machine-Learing-Model-Deployment-in-Amazon-SageMaker

In this Repositer, you learn how to use `Amazon SageMaker` to **build, train, and deploy a machine learning (ML) model using the XGBoost ML algorithm**. Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly.
Taking ML models from conceptualization to production is typically complex and time-consuming. You have to manage large amounts of data to train the model, choose the best algorithm for training it, manage the compute capacity while training it, and then deploy the model into a production environment. Amazon SageMaker reduces this complexity by making it much easier to build and deploy ML models. After you choose the right algorithms and frameworks from the wide range of choices available, SageMaker manages all of the underlying infrastructure to train your model at petabyte scale, and deploy it to production.

#### Problem Stastement:
* The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).
* you will assume the role of a machine learning developer working at a bank. You have been asked to develop a machine learning model to predict whether a customer will enroll for a certificate of deposit (CD).
* The model will be trained on the Bank Marketing Data Set that contains information on customer demographics, responses to marketing events, and external factors. The data has been labeled for your convenience, and a column in the dataset identifies whether the customer is enrolled for a product offered by the bank. A version of this dataset is publicly available from the Machine Learning Repository curated by the University of California, Irvine.
---
### Attribute Information:

Input variables:
**bank client data:**
1. age (numeric)
2. job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5. default: has credit in default? (categorical: 'no','yes','unknown')
6. housing: has housing loan? (categorical: 'no','yes','unknown')
7. loan: has personal loan? (categorical: 'no','yes','unknown')
**related with the last contact of the current campaign:**
8. contact: contact communication type (categorical: 'cellular','telephone')
9. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
**other attributes:**
12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. previous: number of contacts performed before this campaign and for this client (numeric)
15. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
**social and economic context attributes**
16. emp.var.rate: employment variation rate - quarterly indicator (numeric)
17. cons.price.idx: consumer price index - monthly indicator (numeric)
18. cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19. euribor3m: euribor 3 month rate - daily indicator (numeric)
20. nr.employed: number of employees - quarterly indicator (numeric)

**Output variable (desired target):**
21. y - has the client subscribed a term deposit? (binary: 'yes','no')


[Bank Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/)
---

### In this tutorial, you learn how to:

1. Create a SageMaker notebook instance
2. Prepare the data
3. Train the model to learn from the data
4. Deploy the model
5. Evaluate your ML model's performance

![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/2021-02-13_10-07-00.png)

### Step 1. Create an Amazon SageMaker notebook instance for data preparation
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/2021-02-13_10-11-24.png)
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/2021-02-13_10-12-58.png)
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/2021-02-13_10-15-42.png)
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/2021-02-13_10-16-13.png)
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/2021-02-13_10-16-35.png)

### Step 2. Prepare the data
In this step, you use your Amazon SageMaker notebook instance to preprocess the data that you need to train your machine learning model and then upload the data to Amazon S3.
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/2021-02-13_10-21-16.png)
c. In a new code cell on your Jupyter notebook, copy and paste the following code and choose Run.

This code imports the required libraries and defines the environment variables you need to prepare the data, train the ML model, and deploy the ML model.
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/tutorial-sagemaker-import-libraries..png)
d. Create the S3 bucket to store your data. Copy and paste the following code into the next code cell and choose Run.

Note: Make sure to replace the bucket_name your-s3-bucket-name with a unique S3 bucket name. If you don't receive a success message after running the code, change the bucket name and try again.

![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/tutorial-sagemaker-create-bucket.png)

e. Download the data to your SageMaker instance and load the data into a dataframe. Copy and paste the following code into the next code cell and choose Run.
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/tutorial-sagemaker-download-data.png)

f. Shuffle and split the data into training data and test data. Copy and paste the following code into the next code cell and choose Run.

The training data (70% of customers) is used during the model training loop. You use gradient-based optimization to iteratively refine the model parameters. Gradient-based optimization is a way to find model parameter values that minimize the model error, using the gradient of the model loss function.

The test data (remaining 30% of customers) is used to evaluate the performance of the model and measure how well the trained model generalizes to unseen data.
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/tutorial-sagemaker-shuffle-split-data.png)

### Step 3. Train the ML model

In this step, you use your training dataset to train your machine learning model.
a. In a new code cell on your Jupyter notebook, copy and paste the following code and choose Run.

This code reformats the header and first column of the training data and then loads the data from the S3 bucket. This step is required to use the Amazon SageMaker pre-built XGBoost algorithm.
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/tutorial-sagemaker-reformat-data.png)

b. Set up the Amazon SageMaker session, create an instance of the XGBoost model (an estimator), and define the model’s hyperparameters. Copy and paste the following code into the next code cell and choose Run.
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/tutorial-sagemaker-create-session.png)

c. Start the training job. Copy and paste the following code into the next code cell and choose Run.

This code trains the model using gradient optimization on a ml.m4.xlarge instance. After a few minutes, you should see the training logs being generated in your Jupyter notebook.
![](https://github.com/reddyprasade/Machine-Learing-Model-Deployment-in-Amazon-SageMaker/blob/main/img/tutorial-sagemaker-train-model.png)

