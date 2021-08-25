## Big Data
The initial definition of Big Data revolves around the three V's:  
**Volume, Velocity, and Variety**  
**Volume**: the sample size is too large to store it in one machine. There are some techniques such as MapReduce, Hadoop, etc.  
**Velocity**: data is generated and collected very fast, so we need efficient computational algorithms to analyze the data on the fly.  
**Variety**: the data types might take different shapes, such as images, videos, etc.  

## High-Dimensional Data
High-Dimensional data can be defined as data set with a large number of attributes, such as images, videos, surveys, etc. Our main question is **_How we can extract useful information from these massive data sets?_**  
HD analytics challenge is mainly related to the **Curse of dimensionality**:
* **_Model learning issue_**: as distance between observations increases with the dimensions, the sample size required for learning a model drastically increases.
  * Solutions: feature extraction and dimension reduction through low-dimensional learning.  

## Functional Data
Functional data can be defined as a fluctuating quantity or impulse whose variations represent information and is often represented as a function of time or space.
