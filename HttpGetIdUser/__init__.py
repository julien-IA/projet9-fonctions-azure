import azure.functions as func
import pandas as pd
from io import StringIO, BytesIO
import json
import numpy as np

def main(req: func.HttpRequest, clickssamplecsv: bytes) -> func.HttpResponse:

    clicks_sample = pd.read_csv(StringIO(clickssamplecsv))
    liste_id = clicks_sample['user_id'].unique().tolist()
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    return func.HttpResponse(
            json.dumps(liste_id, cls=NpEncoder),
             status_code=200
        )
