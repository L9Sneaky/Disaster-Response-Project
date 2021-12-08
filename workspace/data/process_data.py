import sys
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = "id")
    return df
#     pass


def clean_data(df):
    cat = df['categories'].str.split(';',expand=True)
    w = [x[0] for x in cat.iloc[1,:].str.split('-')[:]]
    cat.columns = w
#     cat['related']
    for i in w:
        cat[i] = [x[1] for x in cat[i].str.split('-')[:]]
    df.drop('categories', axis=1, inplace=True)

    df = pd.concat([df, cat],join='inner', axis=1)
    # drop duplicates
    df =df.drop_duplicates(subset=['id'])
    return df
#     pass


def save_data(df, database_filename):
    from sqlalchemy import create_engine

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('InsertTableName', engine, index=False)
#     pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()