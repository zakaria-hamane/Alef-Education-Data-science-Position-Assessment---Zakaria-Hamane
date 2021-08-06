# Grade Prediction


#I- Goal

The purpose of this ML model is to take a problem description, a question, a student's answer,
and whether or not the response is correct, correct, contradictory, or incorrect.
**We will not be required to provide references for your answers**. They're in the application
memory because it only functions in the context of prior asked questions
, this repository is designed as a set of a machine learning system designed to evaluate students answers.

The core features of this project are:
* Data pre-processing (train, test, references)
* Feature engineering based on text similarities
* ML model training and evaluation
* A Web-based demo that runs the ML model via a Flask REST API


#II- Install
The Python libreries required to run the scripts in this repository are listed in the requirements.txt file.
We recommend using Python= 3.8 and a virtualenv. 

1- Before installing the essential packages, upgrade your Python package manager pip:
```
$ pip install --upgrade pip
```

To build a virtual environment, run the following command at the terminal.
```
$ virtualenv dt-grade
```

Activate the virtual environment: 
```
$ source dt-grade/bin/activate
```

Install the necessary python packages:
```
$ pip3 install -r requirements.txt
```


#III- Data
Data comes from *Alef Education* for the *Technical Assessment for Data science Position*
The DT-Grade corpus is made up of brief built answers collected from tutorial dialogues between students and "DeepTutor" a cutting-edge conversational Intelligent Tutoring System (ITS) (Ruset al., 2013; Rus et al., 2015).
The veracity of the student responses in the particular context was noted, as well as if the contextual information was helpful.
There are 900 responses in the dataset (of which about 25 percent required contextual information to properly interpret them).

#IV- Pre-processing phase
The original data (grade data.xml) is in XML format.
Split the data into ten percent "references", "seventy percent" train, and "twenty percent" test sets using the pre-processing script.
In the "references" set, the script generates a numerical representation of all reference answers and student replies.
To construct an embedding vector for each answer, we utilize the Gensim Python library in conjunction with general word embeddings that were pre-trained on Google News.
**You must first download and store the Google News word embeddings in data/ folder**:

```
$ wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
$ gunzip -k -v data/GoogleNews-vectors-negative300.bin.gz
```

Execute the following pre-processing script:
```
$ python3 preprocess.py
```
The data files will be created in the /generatedData directory.
* train.csv: contains students answers for *training* and *cross-validation*
* test.csv: students answers for model evaluation
* references.txt: reference instances with pre-computed embeddings

> Note  
A hash key was generated in the references text file to identify each problem and question.
This will aid in the lookup when classifying a new instance based on its resemblance to references from the same question.


#V- Model Training and Evaluation Phase
A jupyter notebook contains all of the stages for feature engineering, model training, and evaluation.
**Training and Evaluation.ipynb**  

Using the kernel method, I train a multinomial logistic regression (features in the softmax regression are similarity scores between instances and pre-defined references.)
Kernel approaches, in my opinion, can be useful in situations where the number of observations in the training set is low.
The idea is that by comparing a new student answer to reference answers and other previously evaluated student answers, we can aid in the decision-making process.
To maximize the amount of observations used for training, we penalize the model using a 10-fold cross-validation technique.
The global accuracy and the F1-score for each class are used to assess the system's performance.


#VI- Deployment phase webapp demo
You may put the prediction to the test by using the following web app:
```
$ python3 app.py
```
You need to copy and paste the following text items:
* "Problem description" as written in the original data
* "Question" as written in the original data
* "Student answer"

If the problem description and questions do not fall within the scope of References, the model will predict "Unknown" (never seen before).
Otherwise, we obtain a prediction from the list below:
*Correct* | *Correct but incomplete* | *Contradictory* | *Incorrect*

*Note: when executing the app.py script in your terminal wait for the server to run
and access the web app from localhost:port in you browser as shown below.*

![terminal1](https://im.ge/i/hYLOf) ![terminal2](https://im.ge/i/hYUmm)

![webapp1](https://im.ge/i/hYDH1) ![webapp2](https://im.ge/i/hY7yP)