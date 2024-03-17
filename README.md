# KrenoClass3FastAPI

### To setup your environment
1. Open your terminal or on Windows open the Anaconda Terminal
2. Run `conda create --name KrenoC3 python=3.10`
3. Once all the setup is complete then run `conda activate krenoC3`
4. Now go into the boilerplate directory by `cd`
5. And then perform `pip install -r requirements.txt`
6. Once all the libraries are installed
7. Open visual studio code and open this code
8. Open the terminal in VSC
9. In the terminal again run `conda activate krenoC3`
10. Finally run `uvicorn main:app --reload`
11. Now go to your browser and type `http://localhost:8000/`
12. If you see Hello World in JSON then you are good to go

### Data Dictionary

|Name|Sample|Unique Values|Type|
|----|------|-------------|----|
|LoanID|LP001003|370|String|
|Gender|Male|2|String|
|Married|Yes|2|String|
|Dependents|3+|4|String|
|Education|Graduate|2|String|
|ApplicantIncome|4583|312|Float|
|CoapplicantIncome|1508.0|178|Float|
|LoanAmount|128.0|101|Float|
|Loan_Amount_Term|360|10|Float|
|Credit_History|1|2|String|
|Property_Area|Rural|3|String|

### What is the expected output

We want to predict if loan should be passed or rejected. Pass will be 1 and rejected will be 0

