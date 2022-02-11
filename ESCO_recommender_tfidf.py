import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import string

print_top_n = 6 # number of suggestions
occupation_threshold = 0.35 # cosine similarity to use occupations as skill limiters
merge_broader_skills = True  # add broader skills to low-level skills
use_common_model = True # train one model for all data instead of separate ones

# list of test phrases
test_phrases = {'fin':
                    [
                        'ohjelmointia javalla ja pythonilla. Projektinhallintaa ja johtamista.',
                        'olen ollut postinjakajana ja lajittelijana.',
                        "Osaan auttaa pyörätuolin käytössä",
                        "Olen ollut kirjakaupassa harjoittelijana",
                        "Työskentelin ennen päiväkodissa, niin ja koulun keittiöllä",
                        'työskentelen tarjoilijana ravintolassa ja valmistan ruokaa',
                        'asiantuntijaa, joka vastaa projektitoiminnan työkaluista, projektin- ja ohjelmajohtamisprosessista sekä niihin liittyvästä kehittämisestä.'
                    ],
            'eng': ['prepare, deliver and organize mail','helping with wheelchair patients','taking care of small children, feeding, clothing and playing','worked as a waiter in a restaurant, also prepared food and serving it.']
}

# tyomarkkinatori ehdotetut ammatit 'Osaan auttaa pyörätuolin käytössä':
#   koneasentaja, teollisuuskoneet ja -laitteet
#   sähkötekniikan asiantuntija
#   opaskoiran kouluttaja
#   rengasasentaja
#   visuaalisen taiteen opettaja
#   ICT-asiakaspalvelija
#   terveysneuvoja

##---------------------------------------------------------

# load skilldata
DF = pd.read_csv('ESCO_augmented_skilldata.csv',sep=',',index_col=None,dtype=str)
DF=DF.fillna('')
DF_skills = DF
skill_uris = set(DF_skills['conceptUri'])

# parent skills data (one step higher)
DF = pd.read_csv('ESCO_augmented_parents_level1.csv',sep=',',index_col=None,dtype=str)
DF=DF.fillna('')
DF=DF.loc[DF['label_eng'].apply(lambda x:len(x)>4)]
for row in DF.iterrows():
    row[1]['needed_for'] = row[1]['needed_for'].split('|')
    assert len(row[1]['needed_for'])>0
    assert all([x in skill_uris for x in row[1]['needed_for']])
DF_parents = DF

# load occupation data
DF = pd.read_csv('ESCO_augmented_occupations.csv',sep=',',index_col=None,dtype=str)
DF=DF.fillna('')
DF=DF.loc[DF['label_eng'].apply(lambda x:len(x)>4)]
for row in DF.iterrows():
    row[1]['needed_for'] = row[1]['needed_for'].split('|')
    assert len(row[1]['needed_for']) > 0
    assert all([x in skill_uris for x in row[1]['needed_for']])
DF_occupations=DF

# merge parents to other skills
if merge_broader_skills:
    DF_skills = pd.concat([DF_skills,DF_parents])
    DF_skills.reset_index(drop=True,inplace=True)

# print matches
def compare_and_print(sims,y,top_n,utterance,name):
    sims = np.array(sims)
    inds = np.argsort(sims)
    print('TOP-%i similarities of type "%s" for utterance "%s":  ' % (top_n,name,utterance))
    for i, ind in enumerate(range(top_n)):
        print('  %i (%.4f): %s  ' % (i + 1, sims[inds[-(ind + 1)]], y[inds[-(ind + 1)]]))

# remove stop words
def remove_stops(s,stopw):
    s = "".join([x for x in s if (x not in string.punctuation)])
    s = s.split(' ')
    s = [x for x in s if (x.lower() not in stopw)]
    if 0:
        words = set()
        ss = ''
        for x in s:
            if x not in words:
                ss += x + ' '
                words.add(x)
        ss = ss[0:-1]
    else:
        ss = " ".join(s)
    return ss

# text processor
def process_texts(df):
    try:
        X = [list(row[1].values) for row in df.iterrows()]
        X = ['. '.join(x) for x in X]
    except:
        X = df
    X = [remove_stops(x, stop[language]).lower() for x in X]
    #X = [x if len(x) < MAX_LENGTH else x[0:MAX_LENGTH] for x in X]
    return X

stop = {'fin':stopwords.words('finnish'),'eng':stopwords.words('english')}

# run suggestion process
for language in test_phrases.keys():
    print('\n---------------- LANGUAGE = %s -------------------' % (language))
    label = 'label_%s' % language
    alt_label = 'alt_label_%s' % language
    description = 'description_%s' % language

    X_skill = process_texts(DF_skills[[label,alt_label,description]])
    X_occupation = process_texts(DF_occupations[[alt_label,label,description]])
    X_parent = process_texts(DF_parents[[label,alt_label,description]])

    y_skill = list(DF_skills[label])
    y_occupation = list(DF_occupations[label])
    y_parent = list(DF_parents[label])

    if use_common_model:
        tfidf_total = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), max_df=0.1, min_df=1, max_features=None,use_idf=True, smooth_idf=False, norm='l1')
        tfidf_total.fit(X_skill + X_occupation + X_parent) # add all texts
        XX_skill = tfidf_total.transform(X_skill)
        XX_occupation = tfidf_total.transform(X_occupation)
        XX_parent = tfidf_total.transform(X_parent)

    else:
        tfidf_skill = TfidfVectorizer(analyzer='char_wb',ngram_range=(3, 6),max_df=0.1,min_df=1,max_features=None,use_idf=True,smooth_idf=False,norm='l1')
        XX_skill = tfidf_skill.fit_transform(X_skill)

        tfidf_occupation = TfidfVectorizer(analyzer='char_wb',ngram_range=(3, 6),max_df=0.1,min_df=1,max_features=None,use_idf=True,smooth_idf=False,norm='l1')
        XX_occupation = tfidf_occupation.fit_transform(X_occupation)

        tfidf_parent = TfidfVectorizer(analyzer='char_wb',ngram_range=(3, 6),max_df=0.1,min_df=1,max_features=None,use_idf=True,smooth_idf=False,norm='l1')
        XX_parent = tfidf_parent.fit_transform(X_parent)

    # test utterances
    utterances,utterances_raw = process_texts(test_phrases[language]),test_phrases[language]

    for utterance,utterance_raw in zip(utterances,utterances_raw):

        print('')
        if use_common_model:
            yy_skill = tfidf_total.transform([utterance])
            yy_occupation = tfidf_total.transform([utterance])
            yy_parent = tfidf_total.transform([utterance])
        else:
            yy_skill = tfidf_skill.transform([utterance])
            yy_occupation = tfidf_occupation.transform([utterance])
            yy_parent = tfidf_parent.transform([utterance])

        sims_skill = cosine_similarity(XX_skill,yy_skill).flatten()
        sims_occupation = cosine_similarity(XX_occupation, yy_occupation).flatten()
        sims_parent = cosine_similarity(XX_parent,yy_parent).flatten()

        # match skills with occupations
        inds = np.argsort(sims_occupation)
        ind = []
        for k in range(1,len(inds)):
            i = inds[-k:]
            if sims_occupation[i][0]>occupation_threshold:
                for j in DF_occupations.iloc[i]['needed_for'].values[0]:
                    ind.append(DF_skills[DF_skills['conceptUri']==j].index[0])
            else:
                break
        if len(ind)>0: # if above threshold, zero all others
            sims_skill1 = 0 * sims_skill
            sims_skill1[ind] = sims_skill[ind]
        else:
            sims_skill1 = sims_skill # just omit occupations

        #compare_and_print(sims_skill,y_skill,top_n,utterance,'skill')
        compare_and_print(sims_occupation,y_occupation,print_top_n,utterance_raw,'occupations')
        compare_and_print(sims_skill1, y_skill,print_top_n, utterance_raw,'targeted skills')
        #compare_and_print(sims_parent,y_parent,print_top_n, utterance_raw,'parent skills')

print('\nAll done!')
