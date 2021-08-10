import pandas as pd
import os
import re
import numpy as np
from livingTree import SuperTree
from functools import reduce


class Proportions:

    def __init__(self, read_from: str):
        path = read_from
        path_dir = os.listdir(read_from)
        path_dir = [i for i in path_dir if re.match('.*csv', i)]
        path_dir = sorted(path_dir, key=lambda x: int(x.split('-')[1].split('.')[0]))
        # editted by Hui, dropping unknown column when reading dataframes
        estimated_layers = [pd.read_csv(os.path.join(path, i),
                                        index_col=0).drop(columns=['Unknown']).T for i in path_dir]
        self.proportions = []
        self.proportions += [pd.DataFrame(1, index=['root'], columns=estimated_layers[0].columns)]
        self.proportions += estimated_layers

    def __sum_last_layer(self,
                       df: pd.DataFrame
                       ): # summarize to obtain contribution of the last layer
        return df.groupby(df.index.to_series().apply(self.__parent)).sum()

    def __scale_ol(self,
              df,
              df_next
              ): # scale source proportions for one layer

        df_next_sum = self.__sum_last_layer(df_next)
        df_with_children = df.loc[df_next_sum.index, :]
        ratio = df_with_children.div(df_next_sum)
        ratio[ratio > 1] = 1.0
        index = df_next.index
        df_next = df_next.set_index(df_next.index.map(lambda x: ":".join(x.split(":")[:-1])))
        df_next = ratio * df_next
        return df_next.set_index(index)

    def __scale_and_re_assign_ml(self,
                                 dfs): # scale and re-assign proportions for multiple layers
        dfs_scaled = []
        dfs_scaled.append(dfs[0])
        dfs_UNK_assigned = []
        ##------- obtain scaled dataframes
        for i in range(0, len(dfs) - 1):
            df = dfs_scaled[i]  # scale next layer using scaled current layer, not the original one.
            df_next = dfs[i + 1]
            df_next_sum = self.__sum_last_layer(df_next)
            # delete the unknown index
            # format scaled dataframe
            df_next_scaled = self.__scale_ol(df, df_next)
            dfs_scaled.append(df_next_scaled)
        ##-------
        for i in range(0, len(dfs) - 1):
            df = dfs_scaled[i]
            df_next_scaled = dfs_scaled[i + 1]
            #print(df_next_scaled.columns.to_series().value_counts())
            df_next_scaled_sum = self.__sum_last_layer(df_next_scaled)
            df_unknown = self.__calculate_unknown_ol(df, df_next_scaled_sum)  # .round(4)
            df_onelayer = pd.concat([df_next_scaled, df_unknown])
            dfs_UNK_assigned.append(df_onelayer)
        return dfs_UNK_assigned

    def test_across_layers(self,
             props_multilayer: pd.DataFrame
             ): # test consistence of proportions across layers
        for i in range(0, len(props_multilayer) - 1):
            summarized_contrib = self.__sum_last_layer(props_multilayer[i + 1])
            biomes_with_child = summarized_contrib.index
            # maybe problematic
            is_consistent = props_multilayer[i].loc[biomes_with_child, :].eq(summarized_contrib)
            total_elements = summarized_contrib.shape[0] * summarized_contrib.shape[1]
            total_consistence = is_consistent.sum().sum()
            if total_elements == total_consistence:
                print('test passed on layer {}'.format(i + 2))
            else:
                print('test failed on layer {}'.format(i + 2))
                print('The number of elements is', total_elements)
                print('The number of consistent elements is', total_consistence)
                contrib_diff = props_multilayer[i].loc[biomes_with_child, :] - summarized_contrib
                avg_contrib_diff = np.nanmean(contrib_diff[~is_consistent].to_numpy().flatten())
                print('The statistics of contribution difference among all inconsistent results is', avg_contrib_diff)

    @staticmethod
    def __calculate_unknown_ol(df,
                               df_next_sum): # calculate unknown source proportion for one layer
        df_with_children = df.loc[df_next_sum.index.tolist(), :]
        return (df_with_children - df_next_sum).rename(index=lambda x: '{}:Unknown'.format(x))

    def krona_format(self): # format the proportion to visualize using Krona
        dfs_UNK_assigned = self.__scale_and_re_assign_ml(self.proportions)
        self.test_across_layers(dfs_UNK_assigned)
        '''
        for df in dfs_UNK_assigned:
            print(df.columns.to_series().value_counts())
            print(df.columns.to_series().str.count('.').value_counts())
        '''
        all_nodes = reduce(lambda x, y: x + y, [df.index.tolist() for df in dfs_UNK_assigned])
        tree = SuperTree()
        tree.create_node(identifier='root')
        for node in all_nodes:
            tree.create_node(identifier=node, parent=':'.join(node.split(':')[:-1]))
        df_leaves = pd.concat(dfs_UNK_assigned).loc[[node.identifier for node in tree.leaves()], :].T
        #print(df_leaves.index.to_series().value_counts())
        df_leaves = df_leaves.T.drop_duplicates().T
        #print(df_leaves.index.to_series().value_counts())
        return df_leaves

    def export_krona(self, krona_format, output_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for sampleid in krona_format.index:
            sample = krona_format.T.loc[:, [sampleid]]
            environmental_paths = pd.DataFrame(list(map(lambda x: x.split(':'),
                                                        sample.index)), index=sample.index)
            sample = pd.concat([sample, environmental_paths], axis=1)
            #print('columnes:', sample.columns)
            # remove duplicated columns, maybe problematic
            sample = sample.loc[:, ~sample.columns.duplicated()]
            sample.to_csv(os.path.join(output_dir, sampleid + '.tsv'),
                          index=False, header=False, sep='\t', quoting=False)
    
    @staticmethod
    def ggplot_format(self): # format the proportion to visualize using ggplot
        pass
    
    @staticmethod
    def __parent(node_id: str
                 ):
        # get parent node id of the source environment
        return ":".join(node_id.split(":")[:-1])

