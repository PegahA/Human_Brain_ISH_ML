from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
import utils


gene2go = download_ncbi_associations()
objanno = Gene2GoReader(gene2go, taxids=[9606], go2geneids=True)
go2geneIDs = objanno.get_goid2dbids(objanno.associations)
geneID2GO = objanno.get_dbid2goids(objanno.associations)
genes_in_GO = list(geneID2GO.keys())  # these are entrez_ids
goID2goTerm = {item.GO_ID: item.GO_term for item in objanno.associations}


def get_zeng_labels(genes_of_interest, min_genes_per_group=5):
    """Creates a dataframe of binary annotations from Zeng et al. for a list of genes.

    Args:
        genes_of_interest : must be an iterable of entrez_ids
        min_genes_per_group (int, optional): Defaults to 5.

    Returns:
        pd.DataFrame: df w/ entrez_id index, columns are annotations from Zeng et al. with TRUE/FALSE presence values.
    """
    # genes of interest should be a list of entrez_ids
    zeng = utils.prep_zeng()
    zeng = zeng.drop('cortical_marker', axis=1).drop_duplicates(
        subset=['gene_symbol', 'entrez_id'])

    zeng = zeng[zeng.entrez_id.isin(genes_of_interest)]

    all_dummies = []
    for col in ['pattern_description', 'level_description', 'expression_level', 'celltype_markers', 'layer_markers']:
        dummy_cols = pd.get_dummies(zeng.loc[:, col], prefix=col)
        all_dummies.append(dummy_cols)

    zeng_labels = pd.concat(all_dummies, axis=1)
    zeng_labels.index = zeng.entrez_id

    # return zeng_labels[zeng_labels.index.isin(genes_of_interest)]
    cols_to_drop = zeng_labels.sum()[zeng_labels.sum() < min_genes_per_group].index
    zeng_labels.drop(cols_to_drop, axis=1, inplace=True)
    zeng_labels = zeng_labels.add_prefix('zeng_')

    # return as boolean to be consistent with GOdf
    return zeng_labels.astype('bool')


def get_GO_presence_labels(genes_of_interest, min_GO_size=200, max_GO_size=300):
    """Creates a dataframe of GO-group presence for a list of genes.

    Args:
        genes_of_interest : must be iterable of entrez_gene_ids
        min_GO_size (int, optional): Min num of genes in GO group to be included. Defaults to 200.
        max_GO_size (int, optional): Max num of genes in GO group to be included. Defaults to 300.

    Returns:
        pd.DataFrame : df where index is entrezgene, columns are GO group with TRUE/FALSE presence values.
    """
    genes = pd.Series(genes_of_interest)
    go_group_presence = {}

    for GO in go2geneIDs:
        gene_ids = go2geneIDs[GO]

        # boolean vector (length is num of genes in embedding)
        in_go_group_vector = genes.isin(gene_ids)

        if (in_go_group_vector.sum() > min_GO_size) & (in_go_group_vector.sum() < max_GO_size):
            go_group_presence[GO] = in_go_group_vector

    result = pd.DataFrame(go_group_presence)
    result.index = genes
    result.index.name = 'entrezgene'
    return result


def filter_embedding_for_genes_in_GO(embedding, index_type='gene_symbol'):
    """Filters an embedding to only keep rows where genes have an annotation in GO.

    Args:
        embedding (pd.DataFrame): A DataFrame of shape (n_genes, n_dims)
        index_type (str, optional): Defaults to 'gene_symbol'.

    Returns:
        embedding (pd.DataFrame): A DataFrame of shape (n_genes, n_dims)
    """
    gene_entrez_map = pd.read_csv(
        './data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv', usecols=['entrez_id', 'gene_symbol'])
    gene_entrez_map = gene_entrez_map.dropna(
        subset=['entrez_id']).drop_duplicates(subset=['entrez_id'])

    gene_entrez_map = gene_entrez_map[gene_entrez_map.entrez_id.isin(
        genes_in_GO)]

    if index_type == 'gene_symbol':
        return embedding[embedding.index.isin(gene_entrez_map.gene_symbol)]
    else:
        return embedding[embedding.index.isin(gene_entrez_map.entrez_id)]


def filter_embedding_for_genes_in_zeng(embedding, index_type='gene_symbol'):
    """Filters an embedding to keep rows where genes have annotations from Zeng et al. supplement table

    Args:
        embedding (pd.DataFrame): A DataFrame of shape (n_genes, n_dims)
        index_type (str, optional): Defaults to 'gene_symbol'.

    Returns:
        embedding (pd.DataFrame): A DataFrame of shape (n_genes, n_dims)
    """
    zeng_genes = utils.prep_zeng().loc[:, ['gene_symbol', 'entrez_id']]

    if index_type == 'gene_symbol':
        return embedding[embedding.index.isin(zeng_genes.gene_symbol)]
    else:
        return embedding[embedding.index.isin(zeng_genes.entrez_id)]


def merge_embedding_with_GO_labels(emb_df, GO_df):
    """Merges a gene_embedding with GO group presence df.

    Embedding cols are prefixed with emb_, while potential GO presence columns are prefixed with GO:

    Args:
        emb_df (pd.DataFrame): emb_df.index is gene_symbol
        GO_df (pd.DataFrame): GO_df.index is entrezgene

    Returns:
        (pd.DataFrame): Multi-index gene embedding with columns for GO presence concatenated.
    """
    # get df with gene_symbols and entrez_ids from fetal data (more updated than adult probes data)
    all_genes = pd.read_csv(
        './data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv')
    all_genes = all_genes[~((all_genes.gene_symbol.str.startswith('A_')) | (
        all_genes.gene_symbol.str.startswith('CUST_')))].gene_symbol.drop_duplicates()
    all_genes_w_entrez = utils.genesymbols_2_entrezids(all_genes)

    emb_df = emb_df.add_prefix('emb_')
    df = emb_df.merge(all_genes_w_entrez, left_index=True,
                      right_on='gene_symbol')
    df = df.merge(GO_df, left_on='entrez_id', right_index=True)

    return df.set_index(['entrez_id', 'gene_symbol'])


def merge_embedding_with_zeng_labels(emb_df, zeng_df):
    """Merges a gene_embedding with Zeng annotation df

    Embedding cols are prefixed with emb_, while Zeng annotation columns are prefixed with zeng_

    Args:
        emb_df (pd.DataFrame): emb_df.index is gene_symbol
        zeng_df (pd.DataFrame): zeng_df.index is entrezgene

    Returns:
        (pd.DataFrame): Multi-index gene embedding with columns for zeng presence concatenated
    """
    # get df with gene_symbols and entrez_ids from fetal data (more updated than adult probes data)
    all_genes = pd.read_csv(
        './data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv')
    all_genes = all_genes[~((all_genes.gene_symbol.str.startswith('A_')) | (
        all_genes.gene_symbol.str.startswith('CUST_')))].gene_symbol.drop_duplicates()
    all_genes_w_entrez = utils.genesymbols_2_entrezids(all_genes)

    emb_df = emb_df.add_prefix('emb_')
    df = emb_df.merge(all_genes_w_entrez, left_index=True,
                      right_on='gene_symbol')
    df = df.merge(zeng_df, left_on='entrez_id', right_index=True)

    return df.set_index(['entrez_id', 'gene_symbol'])


def genes_w_low_exp():
    zeng = utils.prep_zeng()
    genes_w_low_exp = zeng[zeng.expression_level == '-']

    return genes_w_low_exp.loc[:, ['gene_symbol', 'entrez_id']]


def perform_GOclass_eval(embedding_df,
                         index_type='gene_symbol',
                         min_GO_size=200,
                         max_GO_size=300,
                         n_splits=5,
                         n_jobs=-1):

    if index_type == 'gene_symbol':
        embedding_df = filter_embedding_for_genes_in_GO(
            embedding_df, index_type='gene_symbol')
        entrez_genelist = utils.genesymbols_2_entrezids(embedding_df.index)
        GO_df = get_GO_presence_labels(
            genes_of_interest=entrez_genelist.entrez_id, min_GO_size=min_GO_size, max_GO_size=max_GO_size)

    elif index_type == 'entrez_id':
        embedding_df = filter_embedding_for_genes_in_GO(
            embedding_df, index_type='entrez_id')
        GO_df = get_GO_presence_labels(
            genes_of_interest=embedding_df.index, min_GO_size=min_GO_size, max_GO_size=max_GO_size)
    else:
        raise ValueError(
            "Error: specify index type as either 'gene_symbol' or 'entrez_id'.")

    gene_count_per_GO_group = {col: GO_df[col].sum() for col in GO_df.columns}

    # merge the embedding and GO_df to ensure they have same index
    # returns a multi-index df with gene_symbol and entrez_id
    merged_df = merge_embedding_with_GO_labels(emb_df=embedding_df, GO_df=GO_df)
    X = merged_df.loc[:, merged_df.columns.str.startswith('emb_')]
    y = merged_df.loc[:, merged_df.columns.str.startswith('GO:')]

    print(f'There are {y.shape[1]} GO groups that will be evaluated.')

    GO_SCORES = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    for GOlabel in y:
        print('--'*50)
        print(GOlabel)
        y_GO = y.loc[:, GOlabel]
        GO_term = goID2goTerm[GOlabel]

        f1_scores = []
        auc_scores = []
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y_GO)):
            model = LogisticRegression(penalty='none', n_jobs=n_jobs, max_iter=500)
            X_train = X.iloc[train_idx, :]
            y_train = y_GO.iloc[train_idx]
            X_test = X.iloc[test_idx, :]
            y_test = y_GO.iloc[test_idx]

            model.fit(X_train, y_train)

            # Extract predictions from fitted model
            preds = model.predict(X_test)
            # probs for classes ordered in same manner as model.classes_
            # model.classes_  >>  array([False,  True])
            probas = pd.DataFrame(model.predict_proba(
                X_test), columns=model.classes_)

            # Get metrics for each model
            f1 = f1_score(y_test, preds)
            f1_scores.append(f1)
            auc = roc_auc_score(y_test, probas[True])
            auc_scores.append(auc)

        measures = {'GO_group': GOlabel,
                    'GO_group_title': GO_term,
                    'number_of_used_genes': gene_count_per_GO_group[GOlabel],
                    'f1': np.mean(f1_scores),
                    'AUC': np.mean(auc_scores)}
        GO_SCORES.append(measures)

    return pd.DataFrame(GO_SCORES)


def perform_zeng_eval(embedding_df,
                      index_type='gene_symbol',
                      min_genes_per_group=5,
                      n_splits=5,
                      n_jobs=-1):

    if index_type == 'gene_symbol':
        embedding_df = filter_embedding_for_genes_in_zeng(
            embedding_df, index_type='gene_symbol')
        entrez_genelist = utils.genesymbols_2_entrezids(embedding_df.index)
        print(f'entrez_genelist: {entrez_genelist}')
        zeng_df = get_zeng_labels(genes_of_interest=entrez_genelist.entrez_id, min_genes_per_group=min_genes_per_group)
        print(f'zeng_df shape: {zeng_df.shape}')
    elif index_type == 'entrez_id':
        embedding_df = filter_embedding_for_genes_in_zeng(
            embedding_df, index_type='entrez_id')
        zeng_df = get_zeng_labels(genes_of_interest=embedding_df.index, min_genes_per_group=min_genes_per_group)
    else:
        raise ValueError(
            "Error: specify index type as either 'gene_symbol' or 'entrez_id'.")

    gene_count_per_zeng_annotation = {col: zeng_df[col].sum() for col in zeng_df.columns}
    # merge the embedding and GO_df to ensure they have same index
    # returns a multi-index df with gene_symbol and entrez_id
    # merged_df = merge_embedding_with_GO_labels(emb_df=embedding_df, GO_df=GO_df)
    merged_df = merge_embedding_with_zeng_labels(emb_df=embedding_df, zeng_df=zeng_df)
    print(f'embed_df shape: {embedding_df.shape}')

    print(f'merged df shape: {merged_df.shape}')
    X = merged_df.loc[:, merged_df.columns.str.startswith('emb_')]
    y = merged_df.loc[:, merged_df.columns.str.startswith('zeng_')]
    print(f'There are {y.shape[1]} GO groups that will be evaluated.')

    SCORES = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    for group_label in y:
        print('--'*50)
        print(group_label)
        y_GO = y.loc[:, group_label]

        f1_scores = []
        auc_scores = []
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y_GO)):
            model = LogisticRegression(penalty='none', n_jobs=n_jobs, max_iter=500)
            X_train = X.iloc[train_idx, :]
            y_train = y_GO.iloc[train_idx]
            X_test = X.iloc[test_idx, :]
            y_test = y_GO.iloc[test_idx]

            model.fit(X_train, y_train)

            # Extract predictions from fitted model
            preds = model.predict(X_test)
            # probs for classes ordered in same manner as model.classes_
            # model.classes_  >>  array([False,  True])
            probas = pd.DataFrame(model.predict_proba(
                X_test), columns=model.classes_)

            # Get metrics for each model
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, probas[True])
            f1_scores.append(f1)
            auc_scores.append(auc)
            #print(f"Fold:{i} F1:{f1} AUC:{auc}")

        measures = {'annotation_group': group_label,
                    # 'iteration': i,
                    'number_of_used_genes': gene_count_per_zeng_annotation[group_label],
                    'f1': np.mean(f1_scores),
                    'AUC': np.mean(auc_scores)}
        SCORES.append(measures)

    return pd.DataFrame(SCORES)


def process_embedding(emb_path, output_path, emb_str):
    emb = pd.read_csv(emb_path, index_col='gene_symbol')
    emb.drop('entrez_id', axis=1, inplace=True)

    emb_go_scores = perform_GOclass_eval(emb,
                                         index_type="gene_symbol",
                                         min_GO_size=40,
                                         max_GO_size=200,
                                         n_splits=5,
                                         n_jobs=-1)
    emb_go_scores.to_csv(output_path / f'{emb_str}_GO_results.csv', index=None)

    emb_zeng_scores = perform_zeng_eval(emb)
    emb_zeng_scores.to_csv(output_path / f'{emb_str}_Zeng_results.csv', index=None)


if __name__ == "__main__":
    triplet_embedding_path = Path(
        './data/embeddings/final_model_trained_on_all_cortex_data_with_sz_genes/1603427156_triplet_all_training_embeddings_gene_level_with_info.csv')
    resnet_embedding_path = Path(
        './data/embeddings/cortex_study/plain_resnet/resnet50_embeddings_gene_level_with_info.csv')
    random_embedding_path = Path(
        './data/embeddings/cortex_study/random/random_all_training_embeddings_gene_level_with_info.csv')

    # Embeddings after removing genes with low/no expression
    """
    zeng = utils.prep_zeng()
    genes_w_low_exp = zeng[zeng.expression_level == '-'].loc[:, ['gene_symbol']]
    triplet_embedding = triplet_embedding[~triplet_embedding.index.isin(genes_w_low_exp.gene_symbol)]
    resnet_embedding = resnet_embedding[~resnet_embedding.index.isin(genes_w_low_exp.gene_symbol)]
    random_embedding = random_embedding[~random_embedding.index.isin(genes_w_low_exp.gene_symbol)]
    """

    # test embedding used that has binary measures for each donor to describe whether a gene was assayed
    test_embedding = pd.read_csv('./data/cortex_gene_donor_one_hot_coded.csv', index_col=0)
    test_embedding = test_embedding.set_index('gene_symbol')

    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)

    # Run triplet evaluations
    start = datetime.datetime.now()
    process_embedding(emb_path=triplet_embedding_path,
                      output_path=results_dir,
                      emb_str='triplet')

    process_embedding(emb_path=resnet_embedding_path,
                      output_path=results_dir,
                      emb_str='resnet')

    process_embedding(emb_path=random_embedding_path,
                      output_path=results_dir,
                      emb_str='random')

    end = datetime.datetime.now()
    print(f'Total time taken: {end - start}')
