import hydra
from conf.config import GuideHelperConfig
from extract import extract
from hydra.core.config_store import ConfigStore
from load import load
from settings import DATA_DIR
from transform import transform

cs = ConfigStore.instance()
cs.store(name="GuideHelper", node=GuideHelperConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def pipeline(
       cfg: GuideHelperConfig
):
    data_file_path = f"{DATA_DIR}/{cfg.files.data}"
    metadata_file_path = f"{DATA_DIR}/{cfg.files.metadata}"

    # extract.py -> load and split into pages
    data, metadata = extract(data_file_path, metadata_file_path)
    
    
    # transform.py -> create docs
    docs = transform(data, metadata)

    # load.py -> to vectorstorechunks 
    load(docs)

pipeline()
    
    

