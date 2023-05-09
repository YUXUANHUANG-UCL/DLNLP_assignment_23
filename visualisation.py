import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def visualisation(path):
    data_path = os.path.join(os.path.join(path, 'Datasets'), 'train.csv')
    # read training data
    df_train = pd.read_csv(data_path)
    # show the first five data
    df_train.head()
    
    # check if there are some missing values
    # results: no missing value
    print('Missing value: ')
    print(df_train.isna().sum())
    
    # Data Overview
    sum_df_train = []
    for i in df_train.drop(['text_id','full_text'],axis=1).columns:
        print('Summary of ', i)
        sum_df_train.append(df_train[i].value_counts().sort_index())
        print(sum_df_train)
        
    # Data Visualisation
    if not os.path.exists(os.path.join(path, 'figure')):
        os.mkdir(os.path.join(path, 'figure'))
    fig_path = os.path.join(path, 'figure')
    # Loop over each column in df_train (except 'text_id' and 'full_text')
    if not os.path.exists(os.path.join(os.path.join(path, 'figure'), 'summary')):
        os.mkdir(os.path.join(os.path.join(path, 'figure'), 'summary'))
    for ind, value in enumerate(df_train.drop(['text_id','full_text'],axis=1).columns):
        # Create a new figure
        plt.figure()
        # Create a count plot for the current column
        ax = sns.countplot(data=df_train, x=value)
        # Add a title to the plot
        plt.title('Summary of ' + value)
        # Add gridlines to the plot
        plt.grid(axis='x', color='0.95')
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x()+0.02, p.get_height() + 15))
        # Add a legend to the plot
        plt.legend()
        # Save the plot as a JPG file with a name that includes the column name
        plt.savefig(os.path.join(os.path.join(fig_path, 'summary'), 'sum_' + value + '.jpg'))

    # count mean words
    num_words = []
    for i in df_train['full_text']:
        i = i.split()
        num_words.append(len(i))

    df_train['total_words'] = num_words
    print('Mean of words: ', df_train['total_words'].mean())
    
    if not os.path.exists(os.path.join(os.path.join(path, 'figure'), 'numt')):
        os.mkdir(os.path.join(os.path.join(path, 'figure'), 'numt'))
    for value in df_train.drop(['text_id', 'full_text', 'total_words'],axis=1).columns:
        plt.figure(figsize=(10,8))
        sns.histplot(data=pd.concat([df_train[value], df_train['total_words']], axis=1), hue=value, x='total_words', multiple='stack')
        plt.ylabel('num')
        plt.title('Number - Total Words (' + value + ')')
        plt.savefig(os.path.join(os.path.join(fig_path, 'numt'), 'total_' + value + '.jpg'))
        plt.show()
        plt.close()
        
    plt.figure(figsize=(10,8))
    sns.heatmap(df_train.drop(['text_id', 'full_text'],axis=1).corr(), annot=True, center=0.7, cmap=sns.diverging_palette(20, 220, n=200))
    plt.setp(ax.get_xticklabels(), rotation=30)
    plt.title('Training Set Correlations')
    plt.savefig(os.path.join(fig_path, 'heatmap.jpg'))
    

    x = df_train['full_text']
    y = df_train.drop(['full_text','text_id','total_words'],axis=1)
    '''
    # Split data into 70% train and 30% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Split the remaining 30% into 15% validation and 15% final test
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    '''
    # Calculate the index for the end of the training set
    train_end_idx = int(len(x) * 0.7)

    # Calculate the index for the end of the validation set
    valid_end_idx = int(len(x) * 0.85)

    # Split the data into training, validation, and test sets
    x_train, y_train = x[:train_end_idx], y[:train_end_idx]
    x_valid, y_valid = x[train_end_idx:valid_end_idx], y[train_end_idx:valid_end_idx]
    x_test, y_test = x[valid_end_idx:], y[valid_end_idx:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test