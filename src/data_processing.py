import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_data(dataset):
    """Chuẩn bị dữ liệu từ dataset"""
    train_df = pd.DataFrame(dataset['train'])
    dev_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])

    label_encoder = LabelEncoder()
    all_intents = pd.concat([train_df['intent'], dev_df['intent'], test_df['intent']]).unique()
    label_encoder.fit(all_intents)

    train_df['label'] = label_encoder.transform(train_df['intent'])
    dev_df['label'] = label_encoder.transform(dev_df['intent'])
    test_df['label'] = label_encoder.transform(test_df['intent'])

    return train_df, dev_df, test_df, label_encoder