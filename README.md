# Ambivalence by Design: A Computational Account of Loopholes

This repository accompanies a study of U.S. adults' evalution and engagement of loophole behaviors.

## Environment

Download the [data folder](https://osf.io/hxpez/) and put it in the root of the repository. Create the following directory structure from the root folder:

```
mkdir -p fig
```

## Data analysis and visualization

Analyze and plot data from the utterance evaluation task (Study 1):

```
python src/utterance_eval_data_analysis.py
```

Plot the comparison between model-derived social cost (Study 2) and participants' evaluations
of how much trouble a person will get into for each type of action (compliance, loophole,
non-compliance), taken from [Bridgers et al. (2023).](https://osf.io/preprints/psyarxiv/cnxzv):

```
python src/plot_model_human_comparison_trouble.py
```

Plot the comparison between model predictions and people's judgment of how much a
person understood what the speaker wants (Study 3), given an observation that the listener
performs compliance, loophole, or non-compliance behavior:


```
python src/plot_model_human_comparison_understanding.py
```


Plot the comparison between model-derived predictions and people's judgment of humor (Discussion):


```
python src/plot_model_human_comparison_funny.py
```

## Statistical tests

`stats_analysis` folder contains `R` script for mixed-effect regression analysis.