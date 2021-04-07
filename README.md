
## About this project

This is the final project for a data analysis bootcamp at Ironhack school, Paris in 2021. For this project the task was to find some insights from a collection of various datasets. The coding part of this repo is aggregated from several Jupyter notebooks to give hints on the machine-learning process and is not a full picture of what has been done. Nonetheless, this decoumentation file will fill the gap.

If you just want to take a look at the main main results of the model, please skip this and go to the presentatio. For infos on why and how we conduct this analysis you can refer to the following readme and the code of this repo. There is absolutely no lines related to the visualization of the info, since the main ones are related to the explonatory data analysis not presented here and the other part is very customized to match the design guidelines of the presentation.

[![Presentatio][presentation]](https://www.alanperfettini.com/WAG_Dataset_presentation/#0)

### Project scope
The main limitation of this work comes from the time limit : one working week between the reception of the data and the presentation of the results. The scope of the project was mostly define as an opened question : **what can learn from this data ?**

### First observations
All datasets are inherited from a data aggregating webservice that helps companies to get some visual insights. This is a tricky issue to tackle since we don't have full rights to query a database. First consequence, we can only collect the score or various predefined metrics and the granularity is not enough precise to reach the user level and launch a cluster model.


By chance, we also get a list of all published editorial content, more on this later.


## Defining the needs
First, we need to understand a little bit the context. The datas are colletected from an app which is a side project of a big NGO, a really big one. The whole concept of their app is to help people to take action in their life, real actions to *individually* participate to the current *social* decisive moment regarding the environment issues.

So, there is two broad components in the info structure of the service:
1. Features related to actions : challenges, tasks, etc.
2. Informative content to sensibilize and eventually lead to action

We can be positive that there is a dependancy between these components : the information should lead up to action. Translate this into metric evaluation, the first intuition was to come up with some analytics that can relate to this dependancy. Second, we knew that maybe the most important part of our work was mainly to open some doors and convey the rights recommandations to be abble in the future to get much better data and take look at the same doors.















<!-- MARKDOWN LINKS & IMAGES -->
[presentation]: img/01--presentation.png





