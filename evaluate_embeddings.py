from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from human_ISH_config import *
import pandas as pd



def build_distance_matrix(filename):

    embedding_file_name = EMBED_SET.split(".csv")[0] + "_embeddings.csv"
    embeddings_csv_file = os.path.join(EMBEDDING_DEST, filename, embedding_file_name)

    embed_df = pd.read_csv(embeddings_csv_file)
    image_id_list = embed_df['image_id']
    embed_df = embed_df.set_index(['image_id'])

    dist_df =pd.DataFrame(
        squareform(pdist(embed_df.loc[image_id_list])),
        columns=image_id_list,
        index=image_id_list
    )

    dist_df_file_name = EMBED_SET.split(".csv")[0] + "_dist.csv"
    #dist_df.to_csv(os.path.join(EMBEDDING_DEST, filename, dist_df_file_name))


    return dist_df

