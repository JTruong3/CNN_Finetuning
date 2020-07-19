# CNN_Finetuning
Different fine tuning methods to increase performance in recognition tasks

## Bilinear CNN for fine-grained recognition (ICCV 2015)
http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf

### The pipeline of this method is:
* Take extracted features from two CNN models 
* Combine the extracted features with the use of an outer product multiplication 
* An orderless pooling is used by the summation of the output above
* The output vector descriptor gets passed through a signed square root and normalization step
* Output is then passed through a classification layer for final predictions
<br>
(Note: End-to-end training is possible because the outer product and pooling are both differentiable)

![Bilinear CNN Model](/images/Bilinear_CNN.png)

### Sample Code:
Take feature size as:[Batch,512,Height,Width]
<br>
- **Input:** Features F1:[B,512,16,16] and F2:[B,512,16,16]
- **Outer Product:** X = torch.bmm(F1, torch.transpose(F2, 1, 2)) / (16 ** 2)
- **Pooling:** X = torch.reshape(X, (B, 512 ** 2))
- **Signed Square Root:** X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
- **Normaliztion:** X = torch.nn.functional.normalize(X)
- **Classification function(X)**



