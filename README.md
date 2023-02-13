# Joint Fine Grained Classification along with Features Redundancy Reduction and Invariance Maximization 

### Work:

Using Barlow Twin's Cross-Correlation Loss function between representaions for 
augumentations of same image. Two images are passed to shared extractor model,
features extracted are passed to two networks (projector and classifier) 
seperately and simultanously. Barlow's Twin Loss is applied Feature vectors 
passed through projector network after transformation, and regular CELoss 
is applied on classifier nerwork. Both Loss function is used for training 
simultaneously.

The Method gives noticible accuracy improvement on 
1. FGVC-Aircraft Dataset
2. Stanford Cars Dataset
3. FoodX-251 Dataset


### Repo:

**Train**





Note: Completed as part of CV703 Course work of MBZUAI.