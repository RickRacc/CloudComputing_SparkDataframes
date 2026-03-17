from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, count, avg, max, stddev, expr, desc
import sys

spark = SparkSession.builder.appName("WikiCategorySummary").getOrCreate()

path = "./wiki-categorylinks-small.csv"

if len(sys.argv) > 1:
    path = sys.argv[1]
    print(path)
    
df = spark.read.csv(path, schema="PAGEID string, CATEGORY string")

page_category_counts = df.groupBy("PAGEID").agg(
    count("CATEGORY").alias("num_categories")
)

summary_df = page_category_counts.agg(
    max("num_categories").alias("max_categories"),
    avg("num_categories").alias("avg_categories"),
    expr("percentile_approx(num_categories, 0.5)").alias("median_categories"),
    stddev("num_categories").alias("std_categories")
)

summary_df.show()

top10_categories = (
    df.groupBy("category")
      .agg(countDistinct("PAGEID").alias("num_pages"))
      .orderBy(desc("num_pages"))
      .limit(10)
)

print("Top 10 most used Wikipedia categories:")
top10_categories.show()

pages_using_top10 = (
    df.join(top10_categories.select("category"), on="category", how="inner")
      .select("PAGEID")
      .distinct()
)

page_category_counts = (
    df.groupBy("PAGEID")
      .agg(countDistinct("category").alias("num_categories"))
)


top10_pages = (
    page_category_counts.join(pages_using_top10, on="PAGEID", how="inner")
                        .orderBy(desc("num_categories"), desc("PAGEID"))
                        .limit(10)
)

print("Top 10 pages using the top 10 frequent categories and having the largest number of categories:")
top10_pages.show()