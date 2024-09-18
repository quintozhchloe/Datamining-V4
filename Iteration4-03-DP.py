"""

@author: Chloe huang
"""
# 02 data understanding
from pyspark.sql import SparkSession
from pyspark.sql.functions import round as spark_round, col, count, when, isnan, sum as spark_sum, mean, stddev
from pyspark.sql.types import IntegerType, FloatType, DoubleType
import matplotlib.pyplot as plt



# Initialize Spark session
spark = SparkSession.builder.appName("DataUnderstanding").getOrCreate()

# Load datasets
poverty_data = spark.read.csv('multidimensional_poverty.csv', header=True, inferSchema=True)
#education_data = spark.read.csv('Inequality in Education.csv', header=True, inferSchema=True)
income_data = spark.read.csv('Inequality in Income.csv', header=True, inferSchema=True)
gender_ineq_data = spark.read.csv('gender_inequality.csv', header=True, inferSchema=True)

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
# p_df.show()
p_df.printSchema()
#p_df.describe().show()
# get cols and rows
p_rows = p_df.count()
p_columns = len(p_df.columns)
# show cols and rows
print(f"poverty rows: {p_rows}, poverty cols: {p_columns}")



#i_df.show()
i_df.printSchema()
#i_df.describe().show()
# get cols and rows
i_rows = i_df.count()
i_columns = len(i_df.columns)
# show cols and rows
print(f"income rows: {i_rows},income cols: {i_columns}")

#g_df.show()
g_df.printSchema()
#g_df.describe().show()
# get cols and rows
g_rows = g_df.count()
g_columns = len(g_df.columns)
# show cols and rows
print(f"gender rows: {g_rows},gender cols: {g_columns}")


# Explore Data
p_df.describe().show()
p_df.printSchema()
p_df_desc = p_df.describe()
p_df_desc = p_df_desc.select([spark_round(col(c), 2).alias(c) for c in p_df_desc.columns])
p_df_desc.show()

p_df.select('Health_Deprivation', 'Population_in_Multidimensional_Poverty','Population_Below_National_Poverty_Line', 'Education_Deprivation', 'Living_Standards').describe().show()
p_df.groupBy('Country').count().show()

numeric_cols = [field.name for field in p_df.schema.fields if isinstance(field.dataType, (IntegerType, FloatType, DoubleType))]
numeric_df = p_df.select(numeric_cols)

# Compute correlations
for col1 in numeric_cols:
   for col2 in numeric_cols:
        corr_val = p_df.stat.corr(col1, col2)
        print(f"Correlation between {col1} and {col2}: {corr_val:.2f}")

selected_cols = ['Health_Deprivation', 'Population_in_Multidimensional_Poverty','Population_Below_National_Poverty_Line', 'Education_Deprivation', 'Living_Standards']






pandas_df = p_df.select('Population_Below_National_Poverty_Line', 'Education_Deprivation').toPandas()


plt.figure(figsize=(10, 6))
plt.plot(pandas_df.index, pandas_df['Population_Below_National_Poverty_Line'], label='Population_Below_National_Poverty_Line')
plt.plot(pandas_df.index, pandas_df['Education_Deprivation'], label='Education_Deprivation')
plt.xlabel('Population_Below_National_Poverty_Line')
plt.ylabel('Education_Deprivation')
plt.title('Population Below National Poverty Line and Education Deprivation')
plt.legend()
plt.show()


spark.stop()





#correlations income
i_df.printSchema()
selected_cols = ['HDI Rank (2021)', 'Inequality in income (2010)', 'Inequality in income (2011)', 'Inequality in income (2012)','Inequality in income (2013)','Inequality in income (2014)','Inequality in income (2015)','Inequality in income (2016)','Inequality in income (2017)','Inequality in income (2018)','Inequality in income (2019)','Inequality in income (2020)','Inequality in income (2021)']
i_df_corr = i_df.select(selected_cols)
for col1 in selected_cols:
    for col2 in selected_cols:
        corr_val = i_df_corr.stat.corr(col1, col2)
        print(f"Correlation between {col1} and {col2}: {corr_val:.2f}")




g_df=g_df.withColumn('Gender Inequality Index (GII)', col('Gender Inequality Index (GII)').cast(DoubleType()))
g_df=g_df.withColumn('Adolescent Birth Rate', col('Adolescent Birth Rate').cast(DoubleType()))
g_df=g_df.withColumn('Maternal Mortality Ratio', col('Maternal Mortality Ratio').cast(DoubleType()))
g_df=g_df.withColumn('Percent Representation in Parliament', col('Percent Representation in Parliament').cast(DoubleType()))
g_df=g_df.withColumn('Population with Secondary Education (Female)', col('Population with Secondary Education (Female)').cast(DoubleType()))
g_df=g_df.withColumn('Population with Secondary Education (Male)', col('Population with Secondary Education (Male)').cast(DoubleType()))
g_df=g_df.withColumn('Labour Force Participation Rate (Female)', col('Labour Force Participation Rate (Female)').cast(DoubleType()))
g_df=g_df.withColumn('Labour Force Participation Rate (Male)', col('Labour Force Participation Rate (Male)').cast(DoubleType()))
g_df.printSchema()



#correlations gender
selected_cols = ['Gender Inequality Index (GII)','Maternal Mortality Ratio','Adolescent Birth Rate','Percent Representation in Parliament','Population with Secondary Education (Female)','Population with Secondary Education (Male)','Labour Force Participation Rate (Female)','Labour Force Participation Rate (Male)']
g_df_corr = g_df.select(selected_cols)
for col1 in selected_cols:
    for col2 in selected_cols:
        corr_val = g_df_corr.stat.corr(col1, col2)
        print(f"Correlation between {col1} and {col2}: {corr_val:.2f}")        

# Check for null values/Missing values


#poverty
missing_values = p_df.select([count(when(col(c).isNull(), c)).alias(c) for c in p_df.columns])
missing_values.show()


# income
missing_values_i = income_data.select([count(when(col(c).isNull() | isnan(col(c)), c)).alias(c) for c in income_data.columns])
missing_values_i.show()


# gender
missing_values_g = gender_ineq_data.select([count(when(col(c).isNull() | isnan(col(c)), c)).alias(c) for c in gender_ineq_data.columns])
missing_values_g.show()


#Check for any duplicate 
#poverty
p_rowsduplicate_rows = p_df.count()- p_df.dropDuplicates().count()
print(f"\n p_rowsduplicate_rows:{p_rowsduplicate_rows}" )
#income
i_rowsduplicate_rows = i_df.count()- i_df.dropDuplicates().count()
print(f"\n i_rowsduplicate_rows:{i_rowsduplicate_rows}" )
#gender
g_rowsduplicate_rows = g_df.count()- g_df.dropDuplicates().count()
print(f"\n g_rowsduplicate_rows:{g_rowsduplicate_rows}" )







#3.1 filtering unnecessary cols


#poverty
columns_to_keep_p = ['Country', 'Education_Deprivation', 'Health_Deprivation', 'Living_Standards', 'Population_Below_National_Poverty_Line', 'Population_Below_1_25_per_Day']
p_df_s = poverty_data.select(columns_to_keep_p)
p_df_s.write.csv('Poverty_dataset_selected.csv', header=True, mode='overwrite')


selected_columns_i = ['Country', 'Human Development Groups', 'Inequality in income (2010)', 'Inequality in income (2011)', 'Inequality in income (2012)', 'Inequality in income (2013)', 'Inequality in income (2014)', 'Inequality in income (2015)', 'Inequality in income (2016)', 'Inequality in income (2017)', 'Inequality in income (2018)', 'Inequality in income (2019)', 'Inequality in income (2020)', 'Inequality in income (2021)']
i_df_s = income_data.select(selected_columns_i)
i_df_s.write.csv('Income_dataset_selected.csv', header=True, mode='overwrite')


selected_columns_g = ['Country', 'GII Rank', 'Gender Inequality Index (GII)', 'Population with Secondary Education (Female)', 'Population with Secondary Education (Male)']
g_df_s = gender_ineq_data.select(selected_columns_g)
g_df_s.write.csv('Gender_dataset_selected.csv', header=True, mode='overwrite')













p_df.describe().show()

# Convert object to float
poverty_data = poverty_data.withColumn('Population_Below_National_Poverty_Line', col('Population_Below_National_Poverty_Line').cast(FloatType()))
poverty_data = poverty_data.withColumn('Population_Below_1_25_per_Day', col('Population_Below_1_25_per_Day').cast(FloatType()))

# Fill null values with median
poverty_data_filled = poverty_data.na.fill({'Population_Below_National_Poverty_Line': poverty_data.approxQuantile('Population_Below_National_Poverty_Line', [0.5], 0.25)[0]})
poverty_data_filled = poverty_data_filled.na.fill({'Population_Below_1_25_per_Day': poverty_data.approxQuantile('Population_Below_1_25_per_Day', [0.5], 0.25)[0]})


# Select relevant columns and save
columns_to_keep_p = ['Country', 'Education Deprivation', 'Health Deprivation', 'Living Standards', 'Population_Below_National_Poverty_Line', 'Population_Below_1_25_per_Day']
poverty_data_selected = poverty_data_filled.select(*columns_to_keep_p)
poverty_data_selected.write.mode('overwrite').csv('C:\\Users\\Administrator\\Desktop\\722 Data mining\\Assignment\\A1\\Iteration3\\Poverty_dataset_selected.csv', header=True)

# Merge data
merged_df = gender_ineq_data.join(income_data, "Country", "outer").join(poverty_data_selected, "Country", "outer")
merged_df.write.mode('overwrite').csv('C:\\Users\\Administrator\\Desktop\\722 Data mining\\Assignment\\A4\\merged_dataset.csv', header=True)

# Drop rows with missing values in specific columns
merged_df_cleaned = merged_df.dropna(subset=['Population_Below_National_Poverty_Line', 'Population_Below_1_25_per_Day'], how='all')
merged_df_cleaned.write.mode('overwrite').csv('C:\\Users\\Administrator\\Desktop\\722 Data mining\\Assignment\\A4\\Clean_dataset.csv', header=True)

# Feature engineering
merged_df_cleaned = merged_df_cleaned.withColumn('Poverty Severity', col('Education Deprivation') * col('Health Deprivation') * col('Living Standards'))

columns_of_interest = ['Inequality in income (2010)', 'Inequality in income (2011)', 'Inequality in income (2012)', 'Inequality in income (2013)', 'Inequality in income (2014)', 'Inequality in income (2015)', 'Inequality in income (2016)', 'Inequality in income (2017)', 'Inequality in income (2018)', 'Inequality in income (2019)', 'Inequality in income (2020)', 'Inequality in income (2021)']
for col_name in columns_of_interest:
    merged_df_cleaned = merged_df_cleaned.withColumn(col_name, col(col_name).cast(FloatType()))

merged_df_cleaned = merged_df_cleaned.withColumn('Average_income_inequality', sum([col(col_name) for col_name in columns_of_interest]) / len(columns_of_interest))

merged_df_cleaned = merged_df_cleaned.drop(*columns_of_interest)
merged_df_cleaned.write.mode('overwrite').csv('C:\\Users\\Administrator\\Desktop\\722 Data mining\\Assignment\\A4\\Clean_dataset_New.csv', header=True)

spark.stop()
