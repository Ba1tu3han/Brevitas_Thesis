'WEIGHT_BIT_WIDTH': 1,

'ACT_BIT_WIDTH': 1,

'IN_BIT_WIDTH': 1,

For binary quantized models (W1A1)
model = model.transform(to_hls.InferBinaryMatrixVectorActivation("decoupled"))

MNIST Dataset: https://drive.google.com/drive/folders/1t-aqpq5b9KUV6K1AvHnmuvhp8ALqM_Le?usp=sharing
