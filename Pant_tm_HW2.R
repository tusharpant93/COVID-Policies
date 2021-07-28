setwd("~/Desktop/Text Analytics and Natural Language Processing (NLP)/hult_NLP_student/HW/HW2")

# Libs
library(tm)
library(qdap)
library(wordcloud)
library(RColorBrewer)
library(pbapply)
library(text2vec)

# Options & Functions
options(stringsAsFactors = FALSE)
Sys.setlocale('LC_ALL','C')

tryTolower <- function(x){
  y = NA
  try_error = tryCatch(tolower(x), error = function(e) e)
  if (!inherits(try_error, 'error'))
    y = tolower(x)
  return(y)
}

cleanCorpus<-function(corpus, customStopwords){
  corpus <- tm_map(corpus, content_transformer(qdapRegex::rm_url))
  corpus <- tm_map(corpus, content_transformer(replace_contraction)) 
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(tryTolower))
  corpus <- tm_map(corpus, removeWords, customStopwords)
  return(corpus)
}

# Create custom stop words
stops <- c(stopwords('english'), 'student', 'case', 'score', 'training')

# Read in multiple files as individuals
txtFiles <- list.files(pattern = 'student_tm_case_score_data|student_tm_case_training_data')

for (i in 1:length(txtFiles)){
  assign(txtFiles[i], read.csv(txtFiles[i]))
  cat(paste('read completed:',txtFiles[i],'\n'))
}

# Vector Corpus; omit the meta data
score_data <- VCorpus(VectorSource(student_tm_case_score_data.csv))
training_data     <- VCorpus(VectorSource(student_tm_case_training_data.csv))

# Clean up the data
score_data <- cleanCorpus(score_data, stops)
training_data     <- cleanCorpus(training_data, stops)

# Another way to extract the cleaned text 
score_data <- unlist(pblapply(score_data, content))
training_data     <- unlist(pblapply(training_data, content))

# FYI
length(score_data)

write.csv(score_data,"Pant_TM_score")
# Instead of 2000 individual documents, collapse each into a single "subject" ie a single document
score_data <- paste(score_data, collapse = ' ')
training_data   <- paste(training_data, collapse = ' ')

# FYI pt2
length(score_data)

# Combine the subject documents into a corpus of *2* documents
allData <- c(score_data, training_data)
allData <- VCorpus((VectorSource(allData)))
allData

# Make TDM
dataTDM  <- TermDocumentMatrix(allData)
dataTDMm <- as.matrix(dataTDM)

# Make sure order is correct!
colnames(dataTDMm) <- c('score', 'training')

# Examine
head(dataTDMm)

dataTDM

commonality.cloud(dataTDMm, 
                  max.words=150, 
                  random.order=FALSE,
                  colors='blue',
                  scale=c(3.5,0.25))

dataTDM.lsa <- lw_bintf(dataTDMm)*gw_idf(dataTDMm)
lsaSpace <- lsa(dataTDM.lsa)
aMatrix <- diag(lsaSpace$sk) %*% t(lsaSpace$dk)
simMatrix <- cosine(aMatrix)
simMatrix

# Extract the document LSA values
docVectors <- as.data.frame(lsaSpace$dk)
head(docVectors)

# Construct the Target
yTarget <- c(rep(1,1000), rep(0,1000))

# Append the target var
docVectors$yTarget <- yTarget

# Sample (avoid overfitting)
set.seed(1234)
idx <- sample(1:nrow(docVectors),.6*nrow(docVectors))
training   <- docVectors[idx,]
score <- docVectors[-idx,]

# Fit the model
fit <- glm(yTarget~., training, family = 'binomial')

# Predict in sample
predTraining <- predict(fit, training, type = 'response')
head(predTraining)

# Predict on validation
predValidation <- predict(fit, validation, type = 'response')
head(predValidation)

# Simple Accuracy Eval
yHat <- ifelse(predValidation >= 0.5,1,0)
(confMat <- table(yHat, validation$yTarget))
summary(conf_mat(confMat))
autoplot(conf_mat(confMat))

# End
