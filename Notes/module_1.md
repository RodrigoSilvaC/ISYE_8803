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

## Functional Data Analysis
Functional data can be defined as a fluctuating quantity or impulse whose variations represent information and is often represented as a function of time or space.  

## Regression
Suppose we have a collection of _i.i.d_ training data  

$(x_1, y_1),...,(x_n, y_n)$  

Where x's are explanatory (independent) variables and y is the response (dependent) variable.  
* We want to build a function $f(x)$ to model the relationship between x's and y.  
* An intuitive way of finding $f(x)$ is by minimizing the following loss function  
$min_{f(x)}\sum_{i=1}^n(y_i-f(x_i))^2$  
This loss function represents the squared difference between the observed y and the outcome of the function, $f(x)$.  
* However, minimizing this optimization model is sometimes impossible if we don't know the structure of the function, so we have to impose some constraints/structure on $f(x)$, such as:  
$f(x) = \beta_0+\beta_1x_1+...+\beta_px_p$  

## Splines and Piecewise Regression

### Polynomial Regression  
Polynomial Regression extends the linear model by adding extra predictors, obtained by raising each of the original predictors to a power. It replaces the standard linear model  
$y_i = \beta_0+\beta_1x_i+\epsilon_i$  
with a polynomial function  
$y_i = \beta_0+\beta_1x_i+\beta_2x_i^2+\beta_3x_i^3+...+\beta_dx_i^d+\epsilon_i$
Notice that the coefficients can be easily estimated using least squares linear regression because this is just a standard linear model with predictors $x_i, x_i^2,x_i^3,...,x_i^d$.  

#### Polynomial vs Nonlinear Regression  
* Both types of regressions are used to model a nonlinear response between the response and predictors.
* **Polynomial regression, however, is considered a linear model**. On the other hand, nonlinear regression the function is a nonlinear combination of parameters. **Nonlinearity is not defined by the predictors but the parameters**.
* Nonlinear regression often requires domain knowledge or first principles for finding the underlying nonlinear function.

#### Disadvantages of Polynomial regression
* Remote part of the function is very sensitive to outliers  
![outliers](D:/Cursos/ISYE_8803/Notes/Images/poly_disadvantages.png)  
* Limited flexibility due to its global functional structure. For example, if you use a cubic polynomial function you are assuming that the pattern of observed data follows a cubic form on the whole range of data.  
A solution to these problems could be to move from global regression to local regression. Divide the range of x's that we have into segments and in each segment fit a local polynomial function. This can be formalized with the idea of **splines**.  

### Splines  
Splines can be defined as a linear combination of Piecewise polynomial functions under continuity assumption. The idea is to partition the domain x into continuous intervals and on each interval fit a local polynomial function.   
* Suppose $x \in [a,b]$. Partition the x domain using the following points (a.k.a. **knots**).  
$a< \xi_1 < \xi_2 < ... < \xi_K < b $  
Where $\xi_0 = a$ and $\xi_{K+1} = b$  
* We then fit a polynomial in each interval under the continuity conditions and combine them by the linear combination:   
$f(X) = \sum_{m=1}^K\beta_mh_m(X)$  
