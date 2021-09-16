install.packages('pacman')
library(pacman)
options(scipen = 999) #do not use scientific notation
set.seed(5) #set seed for replicability

### install and load required libraries ###
p_load(tidyverse,
       splines,
       ggplot2, 
       caret)


### read dataset ###
nirs <- read_csv('data/Question3.csv',col_names = FALSE)

### plot mean function ###
mean_fn <- nirs %>% 
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE))) %>% 
  pivot_longer(cols = starts_with('X'),
               names_to = 'x',
               values_to = 'mean') 

mean_fn <-  mean_fn %>% 
  mutate(obser = seq(from = 0, 
                     to = 1, 
                     length.out = 1000))


mean_plot <- ggplot(mean_fn, mapping = aes(x = obser, y = mean)) + 
  geom_line(colour = 'deepskyblue4', size = 1) +
  ggtitle('Mean function of NIR of the soil samples') +
  lims(x = c(0,1),
       y = c(-2,6)) +
  labs(x = NULL,
       y = NULL) +
  scale_x_continuous(breaks=seq(0, 1, 0.2))
  theme_minimal()

mean_plot

### split data ###
mean_fn <- mean_fn %>%
  mutate(id = row_number()) #create id column


train <- mean_fn %>% sample_frac(.80) #training 80%
train <- train %>%
  mutate(id = row_number())

test  <- anti_join(mean_fn, train, by = 'id') #test 20%
test <- test %>%
  mutate(id = row_number())

Y_train <- train$mean
Y_test <- test$mean

### cubic spline ###
results <- tibble()
for (fold in vector) {
  
}
#create vector k with number of knots
k = seq(0,1,length.out = 15)
k = k[2:(length(k)-1)]


#create empty matrix H with columns
h_colnames <- sprintf('h%d', seq(1:(4+length(k)))) #vector with name of columns
H_train <- tibble(.rows = length(train$obser)) #create empty tibble H
H_test <- tibble(.rows = length(test$obser)) #create empty tibble H


#populate basis matrix H_train
i = 1
for (coln in h_colnames) {
    if (coln == 'h1') {
      a <- coln
      H_train <- H_train %>%
        mutate(!!a := rep(1,length(train$id)))
    } else if (coln == 'h2') {
      a <- coln
      H_train <- H_train %>%
        mutate(!!a := train$obser)
    } else if (coln == 'h3') {
      a <- coln
      H_train <- H_train %>%
        mutate(!!a := train$obser^2)
    } else if (coln == 'h4') {
      a <- coln
      H_train <- H_train %>%
        mutate(!!a := train$obser^3)
    } else {
      a <- coln
      H_train <- H_train %>%
        mutate(!!a := (train$obser-k[i])^3)
      i = i+1
    }
  }

H_train[H_train<= 0] = 0
H_train <- as.matrix(H_train)

##populate basis matrix H_test
i = 1
for (coln in h_colnames) {
  if (coln == 'h1') {
    a <- coln
    H_test <- H_test %>%
      mutate(!!a := rep(1,length(test$id)))
  } else if (coln == 'h2') {
    a <- coln
    H_test <- H_test %>%
      mutate(!!a := test$obser)
  } else if (coln == 'h3') {
    a <- coln
    H_test <- H_test %>%
      mutate(!!a := test$obser^2)
  } else if (coln == 'h4') {
    a <- coln
    H_test <- H_test %>%
      mutate(!!a := test$obser^3)
  } else {
    a <- coln
    H_test <- H_test %>%
      mutate(!!a := (test$obser-k[i])^3)
    i = i+1
  }
}

H_test[H_test<= 0] = 0
H_test <- as.matrix(H_test)

#Least square estimates
B=solve(t(H_train)%*%H_train)%*%t(H_train)%*%Y_train

test$prediction <- H_test%*%B

mean((test$mean - test$prediction)^2)

folds <- createFolds(Y_train, k = 5)
params <- c(5:50)
results <- tibble()

for (param in params) {
  mean_MSE = 0
  for (fold in folds) {
    #create vector k with number of knots
    f_test <- 
    k = seq(0,1,length.out = param)
    k = k[2:(length(k)-1)]
    
    
    #create empty matrix H with columns
    h_colnames <- sprintf('h%d', seq(1:(4+length(k)))) #vector with name of columns
    H_train <- tibble(.rows = length(train$obser)) #create empty tibble H
    H_test <- tibble(.rows = length(test$obser)) #create empty tibble H
    
    
    #populate basis matrix H_train
    i = 1
    for (coln in h_colnames) {
      if (coln == 'h1') {
        a <- coln
        H_train <- H_train %>%
          mutate(!!a := rep(1,length(train$id)))
      } else if (coln == 'h2') {
        a <- coln
        H_train <- H_train %>%
          mutate(!!a := train$obser)
      } else if (coln == 'h3') {
        a <- coln
        H_train <- H_train %>%
          mutate(!!a := train$obser^2)
      } else if (coln == 'h4') {
        a <- coln
        H_train <- H_train %>%
          mutate(!!a := train$obser^3)
      } else {
        a <- coln
        H_train <- H_train %>%
          mutate(!!a := (train$obser-k[i])^3)
        i = i+1
      }
    }
    
    H_train[H_train<= 0] = 0
    H_train <- as.matrix(H_train)
    
    ##populate basis matrix H_test
    i = 1
    for (coln in h_colnames) {
      if (coln == 'h1') {
        a <- coln
        H_test <- H_test %>%
          mutate(!!a := rep(1,length(test$id)))
      } else if (coln == 'h2') {
        a <- coln
        H_test <- H_test %>%
          mutate(!!a := test$obser)
      } else if (coln == 'h3') {
        a <- coln
        H_test <- H_test %>%
          mutate(!!a := test$obser^2)
      } else if (coln == 'h4') {
        a <- coln
        H_test <- H_test %>%
          mutate(!!a := test$obser^3)
      } else {
        a <- coln
        H_test <- H_test %>%
          mutate(!!a := (test$obser-k[i])^3)
        i = i+1
      }
    }
    
    H_test[H_test<= 0] = 0
    H_test <- as.matrix(H_test)
    
    #Least square estimates
    B=solve(t(H_train)%*%H_train)%*%t(H_train)%*%Y_train
    
    test$prediction <- H_test%*%B
    
    mean_MSE = mean_MSE + mean((test$mean - test$prediction)^2)
  results <- results %>%
    mutate(!!fold := mean_MSE/param)
  }
}
  

folds$Fold1
train[folds$1]


train %>%
  filter(id %in% unlist(folds[1]))

str(unlist(folds[1]))


as.list(folds[1])

folds %>%
  select(-folds[1])
str(folds)


train$id 

p <- ggplot(test, mapping = aes(x = obser, y = mean)) + 
  geom_line(colour = 'deepskyblue4', size = 1) +
  geom_line(aes(y = H_test%*%B), colour = 'red' )
  ggtitle('Mean function of NIR of the soil samples') +
  lims(x = c(0,1),
       y = c(-2,6)) +
  labs(x = NULL,
       y = NULL) +
  scale_x_continuous(breaks=seq(0, 1, 0.2))
theme_minimal()
p


