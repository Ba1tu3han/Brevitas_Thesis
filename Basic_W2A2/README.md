#  Config

'WEIGHT_BIT_WIDTH': 2,

'ACT_BIT_WIDTH': 2,

'IN_BIT_WIDTH': 8,

#  For non-binary Models 

(other than W1A1 for example W2A4)

"Conversion to HLS layers" section in the notebook. The third line should be: 

model = model.transform(to_hls.InferQuantizedMatrixVectorActivation("decoupled"))

#  MNIST Dataset

MNIST Dataset: https://drive.google.com/drive/folders/1t-aqpq5b9KUV6K1AvHnmuvhp8ALqM_Le?usp=sharing
