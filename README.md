# Please add your team members' names here. 

## Team members' names 

1. Student Name: Tony Ruiz

   Student UT EID: ajr5936

2. Student Name: Rakesh Singh

   Student UT EID: rps2439

 ...

##  Course Name: CS378 - Cloud Computing 

##  Unique Number: 53410
    


# RESULTS (running on large dataset)



# Task 1

Top Words in Corpus: [('the', 26451056), ('of', 12507151), ('in', 10807932), ('and', 10758007), ('a', 7988396), ('to', 7681799), ('was', 4588136), ('is', 3746524), ('for', 3145951), ('as', 3066049)]

Word Postions in our Feature Matrix. Last 20 words in 20k positions:  [('erica', 19999), ('aqueous', 19998), ('glamour', 19997), ('rockford', 19996), ('uavs', 19995), ('sizeable', 19994), ('rohan', 19993), ('ami', 19992), ('csa', 19991), ('minerva', 19990), ('ub', 19989), ('yarmouth', 19988), ('ericsson', 19987), ('overt', 19986), ('rida', 19985), ('estadio', 19984), ('corbin', 19983), ('ethnographic', 19982), ('rockers', 19981), ('melancholy', 19980)]

# Task 2# Task 1



# Task 3

# Task 3.1
+--------------+-----------------+-----------------+-----------------+
|max_categories|   avg_categories|median_categories|   std_categories|
+--------------+-----------------+-----------------+-----------------+
|           587|5.566655905819486|                4|5.567163467037963|
+--------------+-----------------+-----------------+-----------------+

# Task 3.2

+--------------------+---------+
|            category|num_pages|
+--------------------+---------+
|     Noindexed_pages|  2586023|
|   All_stub_articles|  2243344|
|WikiProject_Biogr...|  1650535|
|Articles_with_sho...|  1516225|
|Redirects_from_moves|  1489008|
|Unprintworthy_red...|  1412217|
|Coordinates_on_Wi...|  1048235|
|Biography_article...|   970853|
|Stub-Class_biogra...|   939454|
|       Living_people|   938708|
+--------------------+---------+

# Task 3.3
+--------+--------------+
|  PAGEID|num_categories|
+--------+--------------+
|62139339|           385|
|  208350|           252|
|   66191|           246|
|  210021|           245|
|   19856|           240|
|18934647|           237|
|  568942|           230|
|   88805|           228|
|   31898|           227|
|  591972|           225|
+--------+--------------+

# Task 4

# Task 4.1
Removing the English stop words such a, the, are, etc. would likely only minimally change the kNN results, but improve the apparent accuracy. Because stop words are very common in every document, their Inverse Document Frequency is very small and they have little influence on the similarity use in kNN. While it will filter some noise and likely lead to better results its unlikely that it will lead to drastically better clustering.

# Task 4.2
Stemming would moderately improve the results as the same concept (such as game, gaming, games) are treated as the same root word and we would see more similarity between documents with the root word, where previously they could have been not associated as two different words. This would provide more robustness to our model, and likely would improve results though perhaps not to a degree where it is a whole new model.

