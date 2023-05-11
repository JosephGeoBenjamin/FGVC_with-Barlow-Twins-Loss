# Fine Grained Visual Classification with Features Redundancy Reduction and Invariance Maximization Loss

![fgvc-bt-ce](https://github.com/JosephGeoBenjamin/FGVC_with-Barlow-Twins-Loss/releases/download/doc-1/fgvc-withBTloss.png)

## Work:

[Technical Report](https://github.com/JosephGeoBenjamin/FGVC_with-Barlow-Twins-Loss/releases/download/doc-1/FGVC-TechReport-CV703-MBZUAI.pdf)

Using Barlow Twin's Cross-Correlation Loss function between representaions for
augumentations of same image. Two images are passed to shared extractor model,
features extracted are passed to two networks (projector and classifier)
seperately and simultanously. Barlow's Twin Loss is applied Feature vectors
passed through projector network after transformation, and regular CELoss
is applied on classifier nerwork. Both Loss function is used for training
simultaneously.

The Method gives noticible accuracy improvement on validations of
1. FGVC-Aircraft Dataset
2. Stanford Cars Dataset
3. FoodX-251 Dataset


### Repo:

**Train Models**

```
python tasks/bt-cls-train.py --load_json config/bt-cls-train_cfg.json
```

**Get Validation Results**
```
python tasks/validate-results.py --load_json JSON --batch-size 256  --output-path OUTDIR
```


Note: Completed as part of CV703 Course work of MBZUAI.
Most codes are written from scratch up with reference and *taken codes* are cited then asn there.