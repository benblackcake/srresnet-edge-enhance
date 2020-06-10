# Srresnet-Edge-Enhance in Tensorflow

This is an implementation of the paper SRGAN-Resnet using TensorFlow. 
[Experiment]Using the idea of SRResnet to develop a netwok more care about the image edge  

## Usage

### Set up

1. Create the .h5 (Training) format dataset in /done_dataset folder.
2. Create the .h5 (Validation).
3. Create the .h5 (Evaluation).
4. Create Benchmarks for testing.

### Training

Srresnet-Edge-Enhance-MSE
```
python main.py --name srresnet-edge-enhance-mse --content-loss mse 
```

Srresnet-Edge-Enhance-edge_loss
```
python main.py --name srresnet-edge-enhance-edge-loss --content-loss edge_loss 
```
