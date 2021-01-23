from pathlib import Path
import pandas as pd


def prep_zeng():
    zeng = pd.read_excel('./data/raw/Table S2.xlsx', header=[1], usecols=['Gene symbol', 'Entrez Gene ID',
                                                                          'pattern description for Fig 2', 'level description for Fig 2', 'Cortical marker (human)', 'Level'])
    zeng = zeng.loc[:, ['Gene symbol', 'Entrez Gene ID', 'pattern description for Fig 2',
                        'level description for Fig 2', 'Cortical marker (human)', 'Level']]
    zeng.columns = zeng.columns.str.rstrip(' (human)')
    zeng.columns = zeng.columns.str.lower().str.replace(' ', '_')
    zeng.rename(columns={'entrez_gene_id': 'entrez_id',
                         'pattern_description_for_fig_2': 'pattern_description',
                         'level_description_for_fig_2': 'level_description',
                         'level': 'expression_level'}, inplace=True)
    zeng.pattern_description = zeng.pattern_description.astype('category')
    zeng.pattern_description.cat.rename_categories({'ND': 'not determined'}, inplace=True)
    zeng.pattern_description.cat.reorder_categories(
        ['widespread', 'laminar', 'scattered', 'sparse', 'not determined'], inplace=True)

    zeng.level_description = zeng.level_description.astype('category')
    zeng.level_description.cat.reorder_categories(['high', 'medium', 'low'], inplace=True)

    zeng.cortical_marker = zeng.cortical_marker.str.split("[/+]| or ")
    zeng = zeng.explode('cortical_marker')
    zeng.cortical_marker = (zeng.cortical_marker.str.replace("layer( )?", "", regex=True)
                                                .str.replace("[?]", "", regex=True)
                                                .str.replace("4c", "4")
                                                .str.replace("6b", "6")
                                                .str.replace("5a", "5")
                                                .str.replace("VEC", "vascular endothelial cells")
                                                .str.replace("([0-6])", "layer \\1", regex=True))
    zeng.cortical_marker.fillna('no annotation', inplace=True)

    layer_markers = [x for x in list(zeng.cortical_marker.unique()) if 'layer' in str(x)]
    celltype_markers = [x for x in list(zeng.cortical_marker.unique()) if 'layer' not in str(x)]
    to_remove = ['laminar', 'no annotation']
    celltype_markers = [marker for marker in celltype_markers if marker not in to_remove]

    zeng['celltype_markers'] = zeng.cortical_marker.apply(lambda x: x if x in celltype_markers else 'No annotation')
    zeng['layer_markers'] = zeng.cortical_marker.apply(lambda x: x if x in layer_markers else 'No annotation')

    zeng['celltype_markers'] = zeng.celltype_markers.astype('category')
    zeng.celltype_markers.cat.reorder_categories(['No annotation', 'interneuron', 'astrocyte', 'oligodendrocyte',
                                                  'vascular endothelial cells', 'others'], inplace=True)
    zeng['layer_markers'] = zeng.layer_markers.astype('category')
    zeng.layer_markers.cat.reorder_categories(['No annotation', 'layer 1', 'layer 2', 'layer 3',
                                               'layer 4', 'layer 5', 'layer 6'], inplace=True)
    zeng['expression_level'] = zeng.expression_level.astype('category')
    zeng.expression_level.cat.reorder_categories(['+++++', '++++', '+++', '++', '+', '-'], inplace=True)
    return zeng


def genesymbols_2_entrezids(genelist):
    """
    Transform list of gene symbols to entrez_ids and returns a tuple of dataframes with results
    """
    # should check that genelist input does not have 'na' values
    probes_file = pd.read_csv('./data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv',
                              usecols=['gene_symbol', 'entrez_id']).drop_duplicates()
    has_entrez = probes_file[probes_file.gene_symbol.isin(genelist)]
    has_entrez = has_entrez.drop_duplicates().dropna(subset=['entrez_id'])

    return has_entrez


def convert_probe_emb_to_gene_emb(probe_emb):
    """Convert embedding with probe_ids for index to gene symbols by averaging probes for same gene symbol.

    Args:
        probe_emb (DataFrame): embedding with index of probe_ids

    Returns:
        gene_emb (DataFrame): embedding with index of gene_symbols
    """
    all_genes = pd.read_csv('./data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv')

    probe2gene = all_genes[all_genes.probeset_name.isin(probe_emb.index)].loc[:, ['probeset_name', 'gene_symbol']]
    # remove probes for 'A_' and 'CUST_' gene_symbols
    probe2gene = probe2gene[~((probe2gene.gene_symbol.str.startswith('A_')) |
                              (probe2gene.gene_symbol.str.startswith('CUST_')))]

    gene_emb = probe_emb.merge(probe2gene, left_index=True, right_on='probeset_name').drop(
        'probeset_name', axis=1).groupby('gene_symbol').mean()

    return gene_emb.drop('na')
