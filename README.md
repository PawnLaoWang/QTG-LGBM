# QTG-LGBM: A Method employed for Prioritizing QTL Causal Gene in Maize
![QTG-LGBM_summary](imgs/QTG-LGBM_summary.png)

## QTG-LGBM
**QTG-LGBM** is a LightGBM-based approach for mining causal genes associated with quantitative trait loci in maize. LightGBM includes *voting parallel*, *Leaf-wise growth* and *Histogram algorithm* to reduce training time and memory consumption.

The QTG-LGBM publication, titled [*<ins>QTG-LGBM: An Method of Prioritizing Causal Genes in Quantitative Trait Locus of Maize</ins>*] provides more detailed information. We kindly invite you to refer to it.

The online service version of QTG-LGBM is also available for you. Please visit [*http://www.deepcba.com/QTG-LGBM*](http://www.deepcba.com/QTG-LGBM) to find out more information.

### QTG-LGBM Introduction
We built a method called **QTG-LGBM** to predict *maize quantitative trait loci(QTL) candidate causal genes* base on *LightGBM*.

*QTG-LGBM* includes three sequential steps. as described in Sections B, C, and D above.

### Experimental Data Introduction
The experimental data in this study consisted of maize PH, FT, and the TBN.
For more experimental data details, we kindly invite you to refer to the QTG-LGBM publication:  [*<ins>QTG-LGBM: An Method of Prioritizing Causal Genes in Quantitative Trait Locus of Maize</ins>*]

## Environment  
### Package Environment
This project is based on Python 3.8.16. The required environment is as follows:  
|    Package      |    Version  |
|----------------:|-------------|
|    numpy        |    1.22.4   |
|    pandas       |    1.3.4    |
|    lightgbm     |    3.3.1    |
|    scikit-learn |    1.0.1    |  

For more required packages, please refer to the [requirements.txt](requirements.txt) file in this project.

## Questions
If you have any questions, requests, or comments, we kindly invite you to contact us at [wangchuang@webmail.hzau.edu.cn](wangchuang@webmail.hzau.edu.cn), [liujianxiao321@webmail.hzau.edu.cn](liujianxiao321@webmail.hzau.edu.cn).
