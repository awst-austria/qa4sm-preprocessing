import pandas as pd
from ismn.meta import MetaData, MetaVar, Depth
from ismn.custom import CustomMetaReader

class FrmSensorInformationReader(CustomMetaReader):
    """
    Allows passing additional information when collecting the metadata in the
    ismn interface for a specific ismn sensor.
    In this case that the metadata is stored in a csv file structured like:

        network;station;sensor;depth_from;depth_to;value;unit;

    """
    def __init__(self, path):
        self.df = pd.read_csv(path, sep=';')

    def read_metadata(self, meta: MetaData):

        cond = (self.df['network'] == meta['network'].val) & \
               (self.df['station'] == meta['station'].val) & \
               (self.df['instrument'] == meta['instrument'].val) & \
               (self.df['depth_from'] == meta['instrument'].depth_from) & \
               (self.df['depth_to'] == meta['instrument'].depth_to)

        vars = []

        for row in self.df[cond].to_dict('records'):
            for k, v in row.items():
                if k in ['network', 'station', 'instrument',
                         'depth_from', 'depth_to', 'frm_nobs']:
                    continue
                else:
                    vars.append(
                        MetaVar(k, v, depth=Depth(row['depth_from'],
                                                  row['depth_to']))
                    )

        return MetaData(vars)

