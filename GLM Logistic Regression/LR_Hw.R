install.packages("readxl")
install.packages("fastDummies")
install.packages("reshape")
install.packages("dplyr")
install.packages("qcc")

library(readxl)
library(fastDummies)
library(reshape)
library(dplyr)
library(qcc)

reduce_rows <- function(df){
  thresh = 0.05
  df['temp'] = df[1]
  for (i in 1:(nrow(df)-1)) {
    for (j in i+1:nrow(df)) {
      if (j<=nrow(df)){
        if (abs((df[i,2] - df[j,2])) < thresh){
          df[j,3] = df[i,3] 
        }
      }
    }
  }
  return (df)
}

replace_orignal <- function(original_df, reduced_df, column){
  for (i in 1:nrow(original_df)) {
    for (j in 1:nrow(reduced_df)) {
      if(original_df[i,column] == reduced_df[j,column]){
        original_df[i,column] = reduced_df[j,3]
      }
    }
  }
  return(original_df)
}


mydata<-read_excel("eBayAuctions.xls")

data.m <- melt(mydata, measure=c(8))

data.cat <- cast(data.m, Category ~ variable, mean)
data.cur <- cast(data.m, currency ~ variable, mean)
data.dur <- cast(data.m, Duration ~ variable, mean)
data.eDay <- cast(data.m, endDay ~ variable, mean)


data_new_cat = reduce_rows(data.cat)
data_new_cur = reduce_rows(data.cur)
data_new_dur = reduce_rows(data.dur)
data_new_eDay = reduce_rows(data.eDay)


mydata = replace_orignal(mydata, data_new_cat, 'Category')
mydata = replace_orignal(mydata, data_new_cur, 'currency')
mydata = replace_orignal(mydata, data_new_dur, 'Duration')
mydata = replace_orignal(mydata, data_new_eDay, 'endDay')

mydataDummy<-dummy_cols(mydata, select_columns = c("Category", "currency", "endDay", "Duration"), remove_first_dummy = TRUE)
mydataDummy$Category <- NULL
mydataDummy$currency <- NULL
mydataDummy$Duration <- NULL
mydataDummy$endDay <- NULL

set.seed(123)
train_ind <- sample(seq_len(nrow(mydataDummy)), size = floor(0.6*nrow(mydataDummy)))
train <- mydataDummy[train_ind, ]
test <- mydataDummy[-train_ind, ]

fit.all = glm(`Competitive?` ~ ., data = train, family = "binomial")
coeff = fit.all$coefficients
View(coeff)
summary(fit.all)

max_coeff = 0
max_index = 0;
for (i in 2:length(coeff)) {
  if(is.na(coeff[i])) {
    coeff[i] = 0
  }
  if (abs(as.numeric(coeff[i])) > max_coeff) {
    max_coeff = abs(as.numeric(coeff[i]))
    max_index = i
  }
}
max_attribute_name = names(coeff)[max_index]

fit.single = glm(`Competitive?` ~., data=train[c('Competitive?', max_attribute_name)], family = binomial(link='logit'))
summary(fit.single)

coeff_df <- as.data.frame(summary(fit.all)$coefficients)
#View(top_n(coeff_df,4,abs(as.numeric(coeff_df$Estimate))))
desc_order_df = coeff_df[order(-abs(as.numeric(coeff_df$Estimate))),]
top_predictors = c(row.names(head(desc_order_df,4)))
View(top_predictors)

out <- "Competitive?"
top_predictor_formula <- as.formula(paste(out, paste(top_predictors, collapse = " + "), sep = " ~ "))
fit.reduced = glm(top_predictor_formula, data=train, family = binomial(link='logit')) # 4 cat
summary(fit.reduced)

for(i in 1:nrow(desc_order_df)) {
  if(desc_order_df[i, 4] < 0.05) {
    significant_predictors <- c(top_predictors, row.names(desc_order_df)[i])
  }
}
significant_predictors_formula <- as.formula(paste(out, paste(significant_predictors, collapse = " + "), sep = " ~ "))
fit.reduced = glm(significant_predictors_formula, data=train, family = binomial(link='logit'))
summary(fit.reduced) 

anova(fit.reduced, fit.all, test='Chisq')

size=rep(length(train$`Competitive?`), length(train$`Competitive?`))
qcc.overdispersion.test(train$`Competitive?`, size=size, type = "binomial")

