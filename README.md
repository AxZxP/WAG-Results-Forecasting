
## About this project

This is the final project for a data analysis bootcamp at Ironhack school, Paris in 2021. For this project the task was to find some insights from a collection of various datasets. The coding part of this repo is aggregated from several Jupyter notebooks to give hints on the machine-learning process and is not a full picture of what has been done. Nonetheless, this decoumentation file will fill the gap.

### Project scope
The main limitation of this work comes from the time limit : one working week between the reception of the data and the presentation of the results. The scope of the project was mostly define as an opened question : **what can learn from this data ?**

### First observations:
All datasets are inherited from a data aggregating webservice that helps companies to get some visual insights. This is a tricky issue to tackle since we don't have full rights to query a database. First consequence, we can only collect the score or various predefined metrics and the granularity is not enough precise to reach the user level and launch a cluster model.


By chance, we also get a list of all published editorial content, more on this later.


## Defining the needs
First, we need to understand a little bit the context. The datas are colletected from an app which is a side project of a big NGO, a really big one. The whole concept of their app is to help people to take action in their life, real actions to *individually* participate to the current *social* decisive moment regarding the environment issues.

So, there is two broad components in the info structure of the service:
1. Features related to actions : challenges, tasks, etc.
2. Informative content to sensibilize and eventually lead to action

We can be positive that there is a dependancy between these components : the information should lead up to action. Translate this into metric evaluation, the first intuition was to come up with some analytics that can relate to this dependancy.



