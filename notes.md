## Embeddings
- When there is not enough GPU memory to apply embeddings encoding to a dataframe column, use the slow approach of df.iterrows
encode one row at a time
use numpy.vstack to add embeddings
