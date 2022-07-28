import warnings

import ex_data_analysis as eda
from globall import MLModel
from spam_detector import SpamDetector

warnings.filterwarnings('ignore')


def main():
    # create required instances
    detector = SpamDetector(dataset_path='data/spam_kaggle.csv')

    # getting initial datafarame of data
    orginal_df = detector.get_df()

    # cleaning the data
    detector.clean_dataset()
    cleaned_df = detector.get_df()

    # plotting cleaned data
    eda.histogram_plot(cleaned_df, SpamDetector.LABEL_TITLE)
    eda.pie_plot(cleaned_df, SpamDetector.LABEL_TITLE)

    # feature engineering
    fe_label = detector.feature_engineering()
    fe_df = detector.get_df()

    # plotting feature engineered data
    eda.histogram_plot(fe_df, fe_label)

    # splitting data into ham and spam
    detector.ham_spam_splitting()

    # drawing word cloud of spams and hams seperately
    # eda.show_word_cloud(detector.get_data_spam(), 'message', 'Spam Messages Word Cloud')
    # eda.show_word_cloud(detector.get_data_ham(), 'message', 'Ham Messages Word Cloud')

    detector.clean_text_messages()

    detector.remove_stop_words()

    detector.stemming_words()

    detector.create_bag_of_words()

    detector.create_tfidf_model()

    detector.build_model(MLModel.GBM)


if __name__ == '__main__':
    main()



    





    
