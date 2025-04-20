
# © 2024 Jennifer Liang, jennliang127@gmail.com 
# [WIP]
from imports import *

class DataPrep:
    """
    DataPrep class contains methods that are use to:
    1. Data 
    2. EDA process (Exploratory Data Analysis process)
    3. Create training dataset and test dataset
    """

    def __init__(self):
        """
        Create instance of input data, indicate test ratio use
        to generate test dataset and training dataset

        Returns: 
            object: __self__
        """

        
    def split_train_test(self,data,test_ratio):
        """
        split_train_test function generate test dataset and 
        train dataset from input data frame. Generates a randomized 
        index from data length. Split test dataset and train dataset 
        length based on test_ratio input. Where test data has length 
        up to int(len(data)*test_ratio) dataset into test set 
        and training set.

        Arg:
            __self__ (object): instance of input dataframe and test ratio
            data (DataFrame):  Cleared data in pd structure
            test_ratio (float): divided ratio of train dataset and 
                                test dataset
                              

        Return: 
            test_data (DataFrame): Random selected number of data points 
                                   equal to len(data) * test_ratio 
            train_data (DataFrame): Random selected number of data points
                                    equal to len(data) * (1-ratio) 
            
        """
        self.data = data
        self.test_ratio = test_ratio
        shuffled_indices = np.random.permutation(len(self.data))
        use_index = int(len(self.data)*self.test_ratio)
        test_indices = shuffled_indices[:use_index]
        train_indices = shuffled_indices[use_index:]

        self.test_data = self.data.iloc[test_indices]
        self.train_data = self.data.iloc[train_indices]

        return  self.test_data, self.train_data

    def test_set_check(self, identifier):
        """
        Use this function to generate stable test/train split 
        set even when dataset has been updated or refreshed. 
        Where, test_set_check method will compute a hash of each 
        instance's identifier to aviod selection of dataset from 
        previously known training set.

        Args:
            __self__ (object): instance of test ratio
            identifier (string): index use the check selected data


        Returns:
            bool: Check if newly selected dataset belongs to previous selected test data indes

        """
        id_return = crc32(np.int64(identifier)) & 0xffffffff < self.test_ratio * 2**32
        return id_return

    def split_train_test_by_id(self, id_column="index"):
        """
        Prepare traning data set and test data set with given ratio 
    
        Args:
            __self__ (object): instance of test ratio
            id_column (string): key string of index use for identify 

        Returns:
            self.train_data (DataFrame): prepared training data set from cleaned data
            self.test_data (DataFrame): testing data prepared to test fitted model performance

        """
        ids = self.data[id_column]
        in_test_set = ids.apply(lambda id_:self.test_set_check(id_,self.test_ratio))
        self.train_data = self.data.loc[~in_test_set]
        self.test_data = self.data.loc(in_test_set)
        return self.train_data, self.test_data
    
    def display_scores(self,scores):
        """
        Display function to show the characteristics of the evaluated model 
        performance.

        Args:
        scores (array): array of evaluated model performance scores, 
                        such as output from cross_val_score

        Returns:
            Score: model scores
            Mean: Mean scores 
            standard deviation: STD of evaluated scores
        """
        print("Scores", scores)
        print(f"Mean:{scores.mean()} ")
        print(f"Standard deviation:", scores.std())

    def plot_num_feature_corr(self, input_data, predict_field):
        """
        Plot number attribute type 
        """

        num_dtypes = ['int16','int32','int64',
                          'float16','float32','float64']

        numeric = [i for i in input_data.keys() if input_data[i].dtype in num_dtypes]
        
        if len(numeric)%2 == 0:
            use_col = 2
        elif len(numeric)%3 == 0:
            use_col=3

        fig, axs = plt.subplots(ncols=use_col, nrows=int(len(input_data.keys())/3))

        for i, feature in enumerate(list(input_data[numeric])):

            #plt.subplot(len(list(numeric)), use_col, i)
            axs[i%3][int(np.floor(i/3))].scatter(x=feature, y=predict_field, data=input_data, s=1)
                
            axs[i%3][int(np.floor(i/3))].set_xlabel('{}'.format(feature), size=10,labelpad=1)
            axs[i%3][int(np.floor(i/3))].set_ylabel(predict_field, size=10, labelpad=1)
            
            for j in range(2):
                plt.tick_params(axis='x', labelsize=12)
                plt.tick_params(axis='y', labelsize=12)
            
        plt.tight_layout()
        plt.show()

        return numeric

                    