import os
import re
from time import time

import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import GradeXML2DataFrame
from utils import get_hash
from utils import get_reference_answers
from utils import Featurizer


def train_test_reference_split(xml_data):
    """Prepare train, dev and test data from original XML into csv format."""

    # Preprocess all data into a Pandas dataframe
    xml2df = GradeXML2DataFrame(xml_data)
    data = xml2df.todf()
    # Modeling/Test split
    remaining, test = train_test_split(data, test_size=0.2, random_state=22)
    # Train/Dev split
    train, references = train_test_split(
        remaining,
        test_size=0.125,
        random_state=22)
    # Save modeling and test data
    if not os.path.exists('generatedData'):
        os.makedirs('generatedData')
    data.to_csv(os.path.join('generatedData', 'grade_data.csv'), index=False)
    train.to_csv(os.path.join('generatedData', 'train.csv'), index=False)
    references.to_csv(os.path.join('generatedData', 'references.csv'), index=False)
    test.to_csv(os.path.join('generatedData', 'test.csv'), index=False)
    print("INFO: Train, test and references data were created successfully "
          "in generatedData directory.")


def references(reference_references, student_references):
    """Create a text file with word2vec embeddings of student
    and reference answers and their labels.

    Args:
        reference_references: csv to extract reference references
        student_references: csv to extract student references
    """

    # Init timer
    start = time()

    # Read a dataset containing all reference answers
    ra_data = pd.read_csv(reference_references)
    # Create hash keys for problem description and question
    ra_data['pd_hash'] = ra_data['problem_description'].apply(get_hash)
    ra_data['qu_hash'] = ra_data['question'].apply(get_hash)
    # Create a dataframe of reference answers one per row
    ra_data['ra_list'] = ra_data['reference_answers'] \
        .apply(get_reference_answers)
    references_ra = ra_data[['pd_hash', 'qu_hash', 'label', 'ra_list']]
    references_ra = references_ra.explode('ra_list')
    references_ra = references_ra.rename(columns={'ra_list': 'answer'})
    references_ra['label'] = 0  # these are possible correct answers (class 0)
    references_ra = references_ra.drop_duplicates()
    print("INFO: Found {} distinct reference reference answers." \
          .format(len(references_ra)))

    # Create a dataframe of student answers
    sa_data = pd.read_csv(student_references)
    # Create hash keys for problem description and question
    sa_data['pd_hash'] = sa_data['problem_description'].apply(get_hash)
    sa_data['qu_hash'] = sa_data['question'].apply(get_hash)
    references_sa = sa_data[['pd_hash', 'qu_hash', 'label', 'answer']]
    references_sa = references_sa.drop_duplicates()
    print("INFO: Found {} distinct student reference answers." \
          .format(len(references_sa)))

    # Create the references dataframe with distinct answers and their labels
    references = references_ra.append(references_sa).drop_duplicates()

    # Create a featurizer object that converts a phrase into embedding vector
    emb_file = os.path.join('data', 'GoogleNews-vectors-negative300.bin')
    featurizer = Featurizer(emb_file)

    # Save the embeddings and labels to disk
    n_references = 0
    with open(os.path.join('generatedData', 'references.txt'), 'w') as f:
        f.write('pd_hash\tqu_hash\tlabel\tanswer\tembedding\n')
        for i in range(len(references)):
            pd_hash = references.iloc[i]['pd_hash']
            qu_hash = references.iloc[i]['qu_hash']
            label = references.iloc[i]['label']
            answer = references.iloc[i]['answer']
            emb = featurizer.doc2vec(references.iloc[i]['answer'])
            emb_txt = ','.join(map(str, emb))
            if norm(emb) != 0:
                n_references += 1
                f.write("%s\t%s\t%s\t%s\t%s\n"\
                        % (pd_hash, qu_hash, label, answer, emb_txt))
    print('INFO: Generating reference embeddings took %.2f seconds.' \
          % (time() - start))
    print("INFO: Found {} non zero references in total.".format(n_references))


def main():
    train_test_reference_split(os.path.join('data', 'grade_data.xml'))
    references(
        os.path.join('generatedData', 'grade_data.csv'),
        os.path.join('generatedData', 'references.csv')
    )


if __name__ == "__main__":
    main()
