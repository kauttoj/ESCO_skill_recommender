import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import string
import pickle
from nltk.tokenize import word_tokenize

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print_top_n_occupations = 7  # number of suggestions
print_top_n_skills  = 20  # number of suggestions
occupation_threshold = 90  # limit search for skills if above this
skills_threshold = 100 # always include skills if above this
merge_broader_skills = True  # add broader skills to low-level skills

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
#test_phrases['fin'] =test_phrases['fin'][0:2]

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
DF = pd.read_csv('ESCO_augmented_skilldata.csv', sep=',', index_col=None, dtype=str)
DF = DF.fillna('')
DF_skills = DF
skill_uris = set(DF_skills['conceptUri'])

# parent skills data (one step higher)
DF = pd.read_csv('ESCO_augmented_parents_level1.csv', sep=',', index_col=None, dtype=str)
DF = DF.fillna('')
DF = DF.loc[DF['label_eng'].apply(lambda x: len(x) > 1)]
for row in DF.iterrows():
    row[1]['needed_for'] = row[1]['needed_for'].split('|')
    assert len(row[1]['needed_for']) > 0
    assert all([x in skill_uris for x in row[1]['needed_for']])
DF_parents = DF

# load occupation data
DF = pd.read_csv('ESCO_augmented_occupations.csv', sep=',', index_col=None, dtype=str)
DF = DF.fillna('')
DF = DF.loc[DF['label_eng'].apply(lambda x: len(x) > 1)]
for row in DF.iterrows():
    row[1]['needed_for'] = row[1]['needed_for'].split('|')
    assert len(row[1]['needed_for']) > 0
    assert all([x in skill_uris for x in row[1]['needed_for']])
DF_occupations = DF

# merge parents to other skills
if merge_broader_skills:
    DF_skills = pd.concat([DF_skills, DF_parents])
    DF_skills.reset_index(drop=True, inplace=True)

# print matches
def compare_and_print(sims,top_n,utterance,name):
    print('TOP-%i similarities of type "%s" for utterance "%s":  ' % (top_n, name, utterance))
    for i, ind in enumerate(range(top_n)):
        print('  %i (%.4f): %s  ' % (i + 1, sims[i][1],sims[i][0]))

# text processor
def process_texts(df):
    try:
        X = [list(row[1].values) for row in df.iterrows()]
        X = ['. '.join(x) for x in X]
    except:
        X = df
    X = [x for x in X]
    # X = [x if len(x) < MAX_LENGTH else x[0:MAX_LENGTH] for x in X]
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

def set_fixed_candidates(agent, cands):
    agent.opt['use_cuda'] = False
    agent.opt['cap_num_predictions'] = 10 ** 6
    agent.eval_candidates = 'fixed'
    cand_vecs = agent._make_candidate_vecs(cands)
    cand_encs = agent._make_candidate_encs(cand_vecs)
    # a=agent_skills.vectorize(["autan potilaita liikkumisessa ja pöyrätuolin kanssa"])
    # agent_skills._vectorize_text(["autan potilaita liikkumisessa ja pöyrätuolin kanssa"])
    agent.set_fixed_candidates({'fixed_candidates': cands, 'fixed_candidate_vecs': cand_vecs, 'fixed_candidate_encs': cand_encs.cuda(), 'num_fixed_candidates': len(cands)})

def add_scores(dict_old,dict_new):
    for k,val in dict_new.items():
        if k in dict_old:
            dict_old[k] = max(dict_old[k],dict_new[k])
        else:
            dict_old[k] = dict_new[k]
    return dict_old

# run suggestion process
for language in test_phrases.keys():
    print('\n---------------- LANGUAGE = %s -------------------' % (language))

    label = 'label_%s' % language
    alt_label = 'alt_label_%s' % language
    description = 'description_%s' % language

    y_skill = list(DF_skills[label])
    y_occupation = list(DF_occupations[label])
    y_parent = list(DF_parents[label])

    agent_skill = pickle.load(open('parlai_agent_%s.pickle' % language, 'rb'))
    set_fixed_candidates(agent_skill, y_skill)

    agent_occupation = pickle.load(open('parlai_agent_%s.pickle' % language, 'rb'))
    set_fixed_candidates(agent_occupation, y_occupation)

    for orig_utterance in test_phrases[language]:

        print('')

        utterance_sentences = split_to_sentences(orig_utterance) + [orig_utterance]

        skill_scores_total = {}
        occupation_scores_total = {}
        valid_skills = []
        valid_occupations = []

        for utterance in utterance_sentences:

            act = {'text': " ".join(word_tokenize(utterance)),
                   'episode_done': True,
                   'id': 'janne_custom',
                   'eval_candidates': '', 'label_candidates': 'fixed'}
            try:
                agent_skill.self_observe([])
            except:
                pass
            agent_skill.observe(act)
            action = agent_skill.act()
            skill_scores = {action['text_candidates'][i]:float(action['sorted_scores'][i]) for i in range(len(action['text_candidates']))}
            assert all([x in y_skill for x in skill_scores.keys()])

            try:
                agent_occupation.self_observe([])
            except:
                pass
            agent_occupation.observe(act)
            action = agent_occupation.act()
            occupation_scores = {action['text_candidates'][i]:float(action['sorted_scores'][i]) for i in range(len(action['text_candidates']))}
            assert all([x in y_occupation for x in occupation_scores.keys()])

            # match skills with occupations
            if occupation_threshold is not None:
                for k,val in occupation_scores.items():
                    if val > occupation_threshold:
                        valid_occupations.append((k,val))
                        for j in DF_occupations.loc[DF_occupations[label]==k,:]['needed_for'].values[0]:
                            valid_skills.append(DF_skills.loc[DF_skills['conceptUri'] == j,label].values[0])
                    else:
                        break
            valid_skills = list(np.unique(valid_skills))

            skill_scores_total = add_scores(skill_scores_total,skill_scores)
            occupation_scores_total = add_scores(occupation_scores_total, occupation_scores)

        if len(valid_skills) > 0:  # if above threshold, zero all others
            skill_scores_total = {k:val for k,val in skill_scores_total.items() if k in valid_skills or val>skills_threshold}
            #assert len(skill_scores)==len(valid_skills)

        skill_scores = sorted(skill_scores_total.items(),key = lambda x:x[1],reverse=True)
        occupation_scores = sorted(occupation_scores_total.items(),key = lambda x:x[1],reverse=True)

        # compare_and_print(sims_skill,y_skill,top_n,utterance,'skill')
        compare_and_print(occupation_scores, print_top_n_occupations,orig_utterance,'occupations')
        if len(valid_skills)>0:
            compare_and_print(skill_scores, print_top_n_skills,orig_utterance,'targeted skills')
        else:
            compare_and_print(skill_scores, print_top_n_skills,orig_utterance, 'skills')
        # compare_and_print(sims_parent,y_parent,print_top_n, utterance_raw,'parent skills')

print('\nAll done!')
