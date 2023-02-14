""" For Running Validation results from Models
Classifier Network training with Barlow twin Correlation on features
"""

import os, sys, time
import argparse
import json
import random, math
import signal, subprocess

from tqdm.autonotebook import tqdm
import PIL.Image

import torch
from torch import nn, optim
import timm.loss

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
from utilities.metricUtils import MultiClassMetrics

# from algorithms.resnet import ClassifierNet
from algorithms.convnext import ClassifierNet
# from algorithms.inception import ClassifierNet
from algorithms.barlowtwin import BarlowWrapnet, lossCEwithBT


from datacode.classifier_data import SimplifiedLoader


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

##============================= Configure and Setup ============================

cfg = rutl.ObjDict(
dataset = "air", # "air+car", "food", "car"
image_size = 299,

batch_size= 256,
workers= 16,
stratergy = "BARLOW",

feature_extract = "convnext-base", #"convnext-tiny/small/base"
featx_pretrain = None,
featx_dropout = 0.0,
classifier = [512,], #First & Last MLPs will be set in code based on class out of dataset and FeatureExtractor
clsfy_dropout = 0.0,
barlow_projector = [4096, 4096, 4096], # dummy

best_model_path = "bestmodel.pth",
gInferencePath =  "BTInferer-Output/"
)


### ----------------------------------------------------------------------------
### ----- Comment below if Parser not needed
parser = argparse.ArgumentParser(description='Classification task')


parser.add_argument('--load_json', type=str, metavar='JSON',
    help='Load settings from file in json format which override values hard codes in py file.')
parser.add_argument('--dataset', type=str, metavar='PATH',
    help="air or air+car or food as per current assignment")
parser.add_argument('--best-model-path', type=str, metavar='PATH', 
    help="path to the model file (.pth)")
parser.add_argument('--batch-size', type=int, metavar='INT',
    help="Batch size for inference")
parser.add_argument('--workers', type=int, metavar='INT',
    help="CPU workers")
parser.add_argument('--output-path', type=str, metavar='PATH',
    help="path to output logging")


args = parser.parse_args()

if args.load_json:
    with open(args.load_json, 'rt') as f:
        cfg.__dict__.update(json.load(f))

if args.dataset:            cfg.dataset = args.dataset
if args.best_model_path:    cfg.best_model_path = args.best_model_path
if args.batch_size:         cfg.batch_size = args.batch_size
if args.workers:            cfg.workers = args.workers
if args.output_path:        cfg.gInferencePath = args.output_path

print("Starting ....")

### ============================================================================

## Checks and Balances
if cfg.stratergy != "BARLOW":
    raise ValueError("This train file only supports Barlow based training use differnt file other usage")
##------

def getDatasetSelection():

    loaderObj = SimplifiedLoader(cfg.dataset)
    validloader, valid_info = loaderObj.get_data_loader(type_= "valid",
                    batch_size=cfg.batch_size, workers=cfg.workers,
                    augument= "INFER", image_size= cfg.image_size)
    lutl.LOG2DICTXT({"Valid-": valid_info}, cfg.gInferencePath +'/misc.txt', console=False)
    print("Classes Count", len(valid_info["Classes"]))
    return validloader, len(valid_info["Classes"])


def getLossSelection():
    valid_loss = nn.CrossEntropyLoss()
    return valid_loss



def simple_main():

    ### SETUP
    rutl.START_SEED()
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    if os.path.exists(cfg.gInferencePath):
        raise Exception("Output Path already exists, Stopping from overwriting folder")
    if not os.path.exists(cfg.gInferencePath): os.makedirs(cfg.gInferencePath)

    ### DATA ACCESS
    validloader, class_size  = getDatasetSelection()
    cfg.classifier.append(class_size) #Adding last layer of MLP

    ### MODEL, OPTIM
    basemodel = ClassifierNet(cfg).cuda(gpu)
    model = BarlowWrapnet(cfg, basemodel).cuda(gpu)

    v_lossfn = getLossSelection()
    lutl.LOG2TXT(f"Model Parameters:{rutl.count_train_param(basemodel)}", cfg.gInferencePath +'/misc.txt')

    ## Load Checkpoints
    if os.path.exists(cfg.best_model_path):
        model.load_state_dict(torch.load(cfg.best_model_path))
        print("Best Checkpoint loaded Succesfully ....")
    else:
        print("Looking for:", cfg.best_model_path)
        raise Exception("Model File not found in the given directory, Lookig for `bestmodel.pth` ")


    ### MODEL Validation
    start_time = time.time()
    validMetric = MultiClassMetrics(cfg.gInferencePath)

    ## ---- Validation Routine ----
    basemodel.eval()
    with torch.no_grad():
        for img, tgt in tqdm(validloader):
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            # with torch.cuda.amp.autocast():
            pred = basemodel.forward(img)  ## or pred, _ = model.forward(img)
            loss = v_lossfn(pred, tgt)
            ## END with
            validMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss)

    ## Log Metrics
    stats = dict(time=int(time.time() - start_time),
                validloss = validMetric.get_loss(),
                validacc = validMetric.get_accuracy(), )
    lutl.LOG2DICTXT(stats, cfg.gInferencePath+'/valid-stats.txt')

    ## Log detailed validation
    detail_stat = dict( time=int(time.time() - start_time),
                        validreport =  validMetric.get_class_report() )
    lutl.LOG2DICTXT(detail_stat, cfg.gInferencePath+'/validation-details.txt', console=False)
    validMetric.reset(True)


if __name__ == '__main__':
    simple_main()