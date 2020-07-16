# CNN_Finetuning
Different fine tuning methods to increase performance

## Bilinear CNN for fine-grained recognition (ICCV 2015)
http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf
The pipeline of this method is:
-> Two different feature extractors produce outputs
-> Outputs are combined through the use of an outer product multiplication
-> Outputs are then pooled and passed through a classification layer
