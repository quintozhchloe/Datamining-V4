# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:26:43 2024

@author: Chloe huang
"""
# 02 data understanding
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, round as spark_round, corr, count, when, isnan, sum as spark_sum
from pyspark.sql.types import DoubleType

# Initialize Spark session
spark = SparkSession.builder.appName("DataUnderstanding").getOrCreate()

# Load datasets
poverty_data = spark.read.csv('C:\\Users\\Administrator\\Desktop\\722 Data mining\\Assignment\\A4\\multidimensional_poverty.csv', header=True, inferSchema=True)
income_data = spark.read.csv('C:\\Users\\Administrator\\Desktop\\722 Data mining\\Assignment\\A4\\Inequality in Income.csv', header=True, inferSchema=True)
gender_ineq_data = spark.read.csv('C:\\Users\\Administrator\\Desktop\\722 Data mining\\Assignment\\A4\\gender_inequality.csv', header=True, inferSchema=True)

# Renaming columns to avoid issues with special characters and spaces
poverty_data = poverty_data.withColumnRenamed('Multidimensional Poverty Index (MPI, HDRO)', 'MPI_HDRO') \
    .withColumnRenamed('Population Below $1.25 per Day', 'Population_Below_1_25_per_Day') \
    .withColumnRenamed('Year and Survey', 'Year_and_Survey') \
    .withColumnRenamed('MPI HDRO Percent', 'MPI_HDRO_Percent') \
    .withColumnRenamed('Multidimensional Poverty Index (MPI, 2010)', 'MPI_2010') \
    .withColumnRenamed('MPI 2010 Percent', 'MPI_2010_Percent') \
    .withColumnRenamed('Population in Multidimensional Poverty', 'Population_in_Multidimensional_Poverty') \
    .withColumnRenamed('Intensity of Deprivation', 'Intensity_of_Deprivation') \
    .withColumnRenamed('Education Deprivation', 'Education_Deprivation') \
    .withColumnRenamed('Health Deprivation', 'Health_Deprivation') \
    .withColumnRenamed('Living Standards', 'Living_Standards') \
    .withColumnRenamed('Population Below National Poverty Line', 'Population_Below_National_Poverty_Line')

# DataFrame references
p_df = poverty_data
i_df = income_data
g_df = gender_ineq_data

# Show DataFrame information
p_df.show()
p_df.printSchema()
i_df.show()
i_df.printSchema()
g_df.show()
g_df.printSchema()

# Explore Data
p_df.describe().show()

# Calculate total nulls
total_nulls_p = p_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in p_df.columns]).toPandas().sum().sum()
print(total_nulls_p)

# Describe specific columns
p_df.select('Health_Deprivation', 'Population_Below_National_Poverty_Line', 'Education_Deprivation', 'Living_Standards', 'Population_Below_1_25_per_Day').describe().show()
p_df.groupBy('Country').count().show()

# Convert columns to numeric
p_df = p_df.withColumn('Population_Below_National_Poverty_Line', col('Population_Below_National_Poverty_Line').cast(DoubleType()))
p_df = p_df.withColumn('Population_Below_1_25_per_Day', col('Population_Below_1_25_per_Day').cast(DoubleType()))

# Compute correlations
numeric_cols = [field.name for field in p_df.schema.fields if isinstance(field.dataType, DoubleType)]
for col1 in numeric_cols:
    for col2 in numeric_cols:
        corr_val = p_df.stat.corr(col1, col2)
        print(f"Correlation between {col1} and {col2}: {corr_val:.2f}")

# Clean data by replacing ".." with NaN and converting to numeric
p_df = p_df.replace('..', None)
p_df = p_df.select([col(c).cast(DoubleType()) if c in numeric_cols else col(c) for c in p_df.columns])

# Fill NaN with median
median_vals = p_df.approxQuantile(numeric_cols, [0.5], 0.25)
median_dict = {col: median_vals[i][0] for i, col in enumerate(numeric_cols)}
p_df = p_df.fillna(median_dict)

# Gender Inequality Data cleaning and transformation
g_df = g_df.select([col(c).cast(DoubleType()) if c != 'Country' else col(c) for c in g_df.columns])
g_df = g_df.fillna(g_df.approxQuantile(numeric_cols, [0.5], 0.25)[0][0])

# Merge datasets
merged_df = g_df.join(i_df, on='Country', how='outer').join(p_df, on='Country', how='outer')

# Save the merged DataFrame to a new CSV file
merged_df.write.csv('C:\\Users\\Administrator\\Desktop\\722 Data mining\\Assignment\\A4\\merged_dataset_A4.csv', header=True)

# Display the merged DataFrame information
merged_df.show()
merged_df.printSchema()

# Clean merged DataFrame
df_cleaned = merged_df.dropna(subset=['Population_Below_National_Poverty_Line', 'Population_Below_1_25_per_Day'], how='all')
df_cleaned.write.csv('C:\\Users\\Administrator\\Desktop\\722 Data mining\\Assignment\\A4\\Clean_dataset_A4.csv'', header=True)

# Constructing the data
cd = df_cleaned.withColumn('Poverty Severity', col('Education_Deprivation') * col('Health_Deprivation') * col('Living_Standards'))

# Calculate average income inequality
columns_of_interest = [f'Inequality in income ({year})' for year in range(2010, 2022)]
cd = cd.withColumn('Average_income_inequality', spark_sum([col(c) for c in columns_of_interest]) / len(columns_of_interest))

# Drop the old inequality columns
cd = cd.drop(*columns_of_interest)

# Fill Human Development Groups with 'Medium' if null
cd = cd.fillna({'Human Development Groups': 'Medium'})

# Save the final cleaned dataset
cd.write.csv('C:\\Users\\Administrator\\Desktop\\722 Data mining\\Assignment\\A4\\Clean_dataset_New_A4.csv', header=True)



















