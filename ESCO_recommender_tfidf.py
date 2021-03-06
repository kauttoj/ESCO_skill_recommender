import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import string

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print_top_n_occupations = 7  # number of suggestions
print_top_n_skills  = 20  # number of suggestions
occupation_threshold = 0.35 # cosine similarity to use occupations as skill limiters
merge_broader_skills = True  # add broader skills to low-level skills
use_common_model = True # train one model for all data instead of separate ones

long_text1 = 'Kauppaopiston jälkeen olen opiskellut kauppatieteitä yliopistossa, sillä haluan seurata alan kehitystä ja saada lisätietoa. Sen lisäksi ' \
            'olen suorittanut useita markkinointiin ja henkilöstöhallintoon liittyviä kursseja. Puhun sujuvasti englantia; olen ollut sekä vaihto-oppilaana että' \
            ' vuoden harjoittelussa USA:ssa. Olin yrityksen palveluksessa lähes 6 vuotta ja viimeiset 2 vuotta vastasin yrityksen markkinoinnista. ' \
            'Pidin työstäni ja yhdessä alaisteni kanssa onnistuimme lisäämään yrityksen markkinaosuutta. Nyt työttömänä ollessani olen tehnyt provisio-palkkaista myyntityötä. Olen hyvä ja selkeä esiintyjä. Joustavuus ja idearikkaus ovat myös vahvuuksiani.'
long_text2 = 'Haemme timanttiseen tiimiimme henkilöä kokin tehtäviin. Henkilöllä ei tarvitse olla kokin koulutusta, riittää myös halukkuus tehdä ruokia ja leipoa keittiössämme ja olla oma-aloitteinen. Tehtäviin voi kuulua tarvittaessa myös muita kuin kokin työtehtäviä. Työssä vaaditaan hygieniapassi sekä anniskelupassi. Työ alkaa mahdollisimman pian.'

# list of test phrases
test_phrases = {'fin':
                    [long_text1,long_text2,
                     'ohjelmointia javalla ja pythonilla.',
                     'Projektinhallintaa ja johtamista.',
                     'ohjelmointia javalla ja pythonilla. Lisäksi osaan projektinhallintaa ja johtamista.',
                     'olen ollut postinjakajana ja lajittelijana.',
                     "Osaan auttaa pyörätuolin käytössä",
                     "autan potilaita liikkumisessa ja pöyrätuolin kanssa",
                     "Olen ollut kirjakaupassa harjoittelijana",
                     "Työskentelin ennen päiväkodissa, niin ja koulun keittiöllä",
                     "päiväkodissa hoitajana ja ohjaajana, sekä avustin keittiötä",
                     'työskentelen tarjoilijana ravintolassa ja valmistan ruokaa',
                     'asiantuntijaa, joka vastaa projektitoiminnan työkaluista, projektin- ja ohjelmajohtamisprosessista sekä niihin liittyvästä kehittämisestä.',
                     'toimin mallina muotinäytöksissä ja esittelin vaatteita'
                     ],
                'eng': ['prepare, deliver and organize mail', 'helping with wheelchair patients', 'taking care of small children, feeding, clothing and playing', 'worked as a waiter in a restaurant, also prepared food and serving it.']
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
DF=DF.loc[DF['label_eng'].apply(lambda x:len(x)>1)]
for row in DF.iterrows():
    row[1]['needed_for'] = row[1]['needed_for'].split('|')
    assert len(row[1]['needed_for'])>0
    assert all([x in skill_uris for x in row[1]['needed_for']])
DF_parents = DF

# load occupation data
DF = pd.read_csv('ESCO_augmented_occupations.csv',sep=',',index_col=None,dtype=str)
DF=DF.fillna('')
DF=DF.loc[DF['label_eng'].apply(lambda x:len(x)>1)]
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
def compare_and_print(sims,top_n,utterance,name):
    print('TOP-%i similarities of type "%s" for utterance "%s":  ' % (top_n, name, utterance))
    for i, ind in enumerate(range(top_n)):
        print('  %i (%.4f): %s  ' % (i + 1, sims[i][1],sims[i][0]))

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

def split_to_sentences(text):
    def do_split(t,s):
        res = []
        for x in t:
            res+=x.split(s)
        return res
    text = text if isinstance(text,list) else [text]
    text = do_split(text,'.')
    text = do_split(text, '!')
    text = do_split(text, '?')
    text = [x for x in text if len(x)>0]
    return text

def add_scores(dict_old,dict_new):
    for k,val in dict_new.items():
        if k in dict_old:
            dict_old[k] = max(dict_old[k],dict_new[k])
        else:
            dict_old[k] = dict_new[k]
    return dict_old

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

    for orig_utterance in test_phrases[language]:

        print('')

        utterance_sentences = split_to_sentences(orig_utterance)
        utterance_sentences = [orig_utterance] + utterance_sentences

        skill_scores_total = {}
        occupation_scores_total = {}

        for utterance0 in utterance_sentences:
            # test utterances
            utterance, utterance_raw = process_texts([utterance0]), utterance0

            print('')
            if use_common_model:
                yy_skill = tfidf_total.transform(utterance)
                yy_occupation = tfidf_total.transform(utterance)
                yy_parent = tfidf_total.transform(utterance)
            else:
                yy_skill = tfidf_skill.transform(utterance)
                yy_occupation = tfidf_occupation.transform(utterance)
                yy_parent = tfidf_parent.transform(utterance)

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

            skill_scores = {y_skill[i]:sims_skill1[i] for i in range(len(y_skill))}
            occupation_scores = {y_occupation[i]: sims_occupation[i] for i in range(len(y_occupation))}

            skill_scores_total = add_scores(skill_scores_total,skill_scores)
            occupation_scores_total = add_scores(occupation_scores_total, occupation_scores)

        skill_scores = sorted(skill_scores_total.items(),key = lambda x:x[1],reverse=True)
        occupation_scores = sorted(occupation_scores_total.items(),key = lambda x:x[1],reverse=True)

        # compare_and_print(sims_skill,y_skill,top_n,utterance,'skill')
        compare_and_print(occupation_scores, print_top_n_occupations,orig_utterance,'occupations')
        compare_and_print(skill_scores, print_top_n_skills,orig_utterance,'targeted skills')
        #compare_and_print(sims_parent,y_parent,print_top_n, utterance_raw,'parent skills')

print('\nAll done!')
