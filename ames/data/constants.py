from argparse import Namespace


RawData = Namespace(
    parent_folder="RAW_DATA",
    test="TEST_DATA",
    submissions="SUBMISSIONS",
    environment_variable="TRAINING",
    target="SalePrice",
    primary_key="Id",
    random_seed=0,
    train_size=0.8,
    test_size=0.2,
)
