[![Presentation][presentation]](https://www.alanperfettini.com/WAG_Dataset_presentation/#0)

## About this project

This is the final project for a data analysis bootcamp at Ironhack school, Paris in 2021. For this project the task was to find some insights from a collection of various datasets. The coding part of this repo is aggregated from several Jupyter notebooks to give hints on the machine-learning process and is not a full picture of what has been done. Nonetheless, this decoumentation file will fill the gap.

If you just want to take a look at the main main results of the model, please skip this and go to the presentation (clic the top image or go here). For infos on why and how we conduct this analysis you can refer to the following readme and the code of this repo. There is absolutely no lines related to the visualization of the info, since the main ones are related to the explonatory data analysis not presented here and the other part is very customized to match the design guidelines of the presentation.

### Project scope
The main limitation of this work comes from the time limit : one working week between the reception of the data and the presentation of the results. The scope of the project was mostly define as an opened question : **what can we learn from this data ?**

### First observations
All datasets are inherited from a data aggregating webservice that helps companies to get some visual insights. This is a tricky issue to tackle since we don't have full rights to query a database. First consequence, we can only collect the score or various predefined metrics and the granularity is not enough precise to reach the user level and launch a cluster model.


By chance, we also get a list of all published editorial content, more on this later.


## Defining the needs
First, we need to understand a little bit the context : data is related to an app which is a side project of a big NGO, a really BIG one. The whole concept of their app is to help people to take action in their life, real actions to *individually* participate to the current *social* change regarding environmental issues.

So, there is two broad components in the info structure of the service:
1. Features related to actions : challenges, tasks, etc.
2. Informative content to sensibilize and eventually lead to action

We can be positive that there is a dependancy between these components : the information should lead up to action. Translate this into metric evaluation, the first intuition was to come up with some analytics that can relate to this dependancy. Second, we knew that maybe the most important part of our work was mainly to open some doors and convey the best recommandations to be abble, after several adjustments on the data management in the backoffice, come back and take look at the same algos to look for some usefull insights.

## The analysis
### The Evolution of a metric

We choose a meaningful metric : the evolution of the score for the Challenge feature of the app. Since it measures an action that recquire a long term investment it should reflect a good indication of the results of the app.

Since its time series dataset and we want to include editorial strategy as exogenous features, we first need to examine the editorial content. Here we go !

#### Building a taxonomy
Unlike the vast majority of UX designer (my main job) that derive their skills from a graphic background, my main expertise is the information architecture. As such, it's not even a option to think about any kind of NLP to parse the datas related to the editorial planning. So I choose the hard way : parsing carefully the whole database of article (titles and homemade categories) to build a meaningfull taxonomy.

This taxonomy is considered as a work in progress since it's time-consuming to do it properly. But in the end we found something good to prepare the future. Sadly, since it's not fully optimized, the Miscelaneous label (the classic label of an unfinished info structure) is one of the largest collection of articles.

![taxonomy][taxonomy]

Other problem as you can see, two categories are related to content on which we can't obtain any data. Notetheless, this taxonomy should be doing a great job for optimization to come since from a editorial persepective it's very comprehensive and can accommodate sub-systems of labelization without any major issue of mismatching or intersection.


#### Sarimax et Prophet
We then used Sarimax et Prophet to build a model based on auto-regression.

### Cohort analysis : the retention 
To see this Challenge feature from another angle we select a dataset of the retention analysis for a weekly cohort. We set the rate of Week 8 as our dependant variable and the rate for weeks 0 to 4 along with the cohort size as independant variables. The selected model is a linear regression. We obtain a decent adjusted R-squared of 0.72 while validating all assumptions.

### The recommandation
This work lead to a set of recommandations. By trying to work on non-optimized datasets we collect a good amount of insights that are in my opinion more valuable that this analysis.


<!-- MARKDOWN LINKS & IMAGES -->
[presentation]: img/01--presentation.png
[taxonomy]: img/08_top_18_themes.png





