# ESCO_skill_recommender
Experiments with ESCO to suggest skills and occupations in Finnish and English.

The simplest approach is classic TF-IDF that applies hierachical structure of ESCO, i.e., occupation and broad skill are associated with lower-level skills. Despite simplicity, we get surprisingly good results as seen below. This is output of the example.
Here "occupations" are most matching occupations and "targeted skills" most matched skills, either guided by occupations or not (depends on threshold level which is here 0.35). 

```
---------------- LANGUAGE = fin -------------------

TOP-6 similarities of type "occupations" for utterance "ohjelmointia javalla ja pythonilla. Projektinhallintaa ja johtamista.":  
  1 (0.1679): ICT-projektipäällikkö  
  2 (0.1581): ohjelmistokehittäjä  
  3 (0.1499): avustava hankepäällikkö  
  4 (0.1422): ratainsinööri  
  5 (0.1225): projektiassistentti  
  6 (0.0903): projektipäällikkö, putkijohtokuljetusten ympäristövaikutukset  
TOP-6 similarities of type "targeted skills" for utterance "ohjelmointia javalla ja pythonilla. Projektinhallintaa ja johtamista.":  
  1 (0.4927): Python (tietokoneohjelmointi)  
  2 (0.4463): projektinhallinnan periaatteet  
  3 (0.4340): projektinhallinta  
  4 (0.4145): ketterä projektinhallinta  
  5 (0.3516): Lean-projektinhallinta  
  6 (0.2780): projektin konfiguraationhallinta  

TOP-6 similarities of type "occupations" for utterance "olen ollut postinjakajana ja lajittelijana.":  
  1 (0.3866): postinlajittelija  
  2 (0.2289): lajittelija, nahat  
  3 (0.2217): ilmoitusten jakaja  
  4 (0.2050): postinkantaja  
  5 (0.1877): puutavaranlajittelija  
  6 (0.1803): jätteiden lajittelija  
TOP-6 similarities of type "targeted skills" for utterance "olen ollut postinjakajana ja lajittelijana.":  
  1 (0.1571): käyttää postitustietojärjestelmiä  
  2 (0.1277): järjestää postilähetysten toimittaminen  
  3 (0.1223): käsitellä postia  
  4 (0.0659): varmistaa postilähetysten eheys  
  5 (0.0426): erottaa pakettityypit  
  6 (0.0047): pitää kirjaa rahdista  

TOP-6 similarities of type "occupations" for utterance "Osaan auttaa pyörätuolin käytössä":  
  1 (0.1063): polkupyörälähetti  
  2 (0.0872): polkupyöränkokooja  
  3 (0.0862): polkupyörämekaanikko  
  4 (0.0829): pyöröhioja  
  5 (0.0820): moottoripyöräkouluttaja  
  6 (0.0809): insinööri, pyörivät koneet ja laitteet  
TOP-6 similarities of type "targeted skills" for utterance "Osaan auttaa pyörätuolin käytössä":  
  1 (0.2581): neuvoa erikoisvälineiden käytössä päivittäisissä toimissa  
  2 (0.2526): avustaa liikuntarajoitteisia matkustajia  
  3 (0.2458): siirtää potilaita  
  4 (0.2348): erikoisvälineiden käyttö päivittäisissä toimissa  
  5 (0.2325): varmistaa, että ajoneuvoissa on esteettömyyslaitteisto  
  6 (0.2212): antaa esteettömyysratkaisuihin liittyviä neuvoja  

TOP-6 similarities of type "occupations" for utterance "Olen ollut kirjakaupassa harjoittelijana":  
  1 (0.4081): johtaja, kirjakauppa  
  2 (0.2146): harjoittaja  
  3 (0.1923): kirjamyyjä  
  4 (0.1832): harjoittaja, tanssi  
  5 (0.1306): kirjailija  
  6 (0.1198): ostosuunnittelija  
TOP-6 similarities of type "targeted skills" for utterance "Olen ollut kirjakaupassa harjoittelijana":  
  1 (0.0780): myyntitoimet  
  2 (0.0702): kirjallisuuden lajityypit  
  3 (0.0569): pysyä ajan tasalla viimeisimmistä kirjajulkaisuista  
  4 (0.0391): mainostaa uusia kirjajulkaisuja  
  5 (0.0359): laatia hinnoittelustrategioita  
  6 (0.0338): käyttää eri viestintäkanavia  

TOP-6 similarities of type "occupations" for utterance "Työskentelin ennen päiväkodissa, niin ja koulun keittiöllä":  
  1 (0.2508): lastenhoitaja, päiväkoti  
  2 (0.2157): keittiöapulainen  
  3 (0.2143): keittiöasentaja  
  4 (0.1882): keittiöpäällikkö  
  5 (0.1719): keittiöavustaja  
  6 (0.1706): koulunkäynninohjaaja, esiopetus  
TOP-6 similarities of type "targeted skills" for utterance "Työskentelin ennen päiväkodissa, niin ja koulun keittiöllä":  
  1 (0.3681): päiväkodin käytännöt  
  2 (0.2725): varmistaa keittiölaitteiden huolto  
  3 (0.1829): valvoa keittiövarusteiden käyttöä  
  4 (0.1805): avustaa koulun tapahtumien järjestämisessä  
  5 (0.1737): valvoa tilattujen keittiötarvikkeiden saapumista  
  6 (0.1706): peruskoulun ala-asteen käytännöt  
   
TOP-6 similarities of type "occupations" for utterance "työskentelen tarjoilijana ravintolassa ja valmistan ruokaa":  
  1 (0.4153): tarjoilija  
  2 (0.2821): ravintolaisäntä/ravintolaemäntä  
  3 (0.2574): ravintolapäällikkö  
  4 (0.2293): ammatillinen opettaja, hotelli- ja ravintola-ala  
  5 (0.1728): pikaravintolan työntekijä  
  6 (0.1680): pikaravintolan tiiminjohtaja  
TOP-6 similarities of type "targeted skills" for utterance "työskentelen tarjoilijana ravintolassa ja valmistan ruokaa":  
  1 (0.3076): tarjoilla ruokaa pöytiin  
  2 (0.2473): valmistella ravintola palvelukuntoon  
  3 (0.1553): tarjoilla viinejä  
  4 (0.1246): tarjoilla juomia  
  5 (0.1172): ruokalistan ruoat ja juomat  
  6 (0.1129): tarkastaa ruokasalin puhtaus  

TOP-6 similarities of type "occupations" for utterance "asiantuntijaa, joka vastaa projektitoiminnan työkaluista, projektin- ja ohjelmajohtamisprosessista sekä niihin liittyvästä kehittämisestä.":  
  1 (0.2405): ICT-projektipäällikkö  
  2 (0.2001): prosessitekniikan asiantuntija  
  3 (0.1915): projektiassistentti  
  4 (0.1881): kemian prosessitekniikan asiantuntija  
  5 (0.1621): prosessivalvonnan asiantuntija, vaatetusteollisuus  
  6 (0.1613): asiantuntija, seuranta ja arviointi  
TOP-6 similarities of type "targeted skills" for utterance "asiantuntijaa, joka vastaa projektitoiminnan työkaluista, projektin- ja ohjelmajohtamisprosessista sekä niihin liittyvästä kehittämisestä.":  
  1 (0.3513): projektin konfiguraationhallinta  
  2 (0.3358): projektinhallinnan periaatteet  
  3 (0.3327): projektiolaitteet  
  4 (0.3051): projektinhallinta  
  5 (0.2983): luoda projektimääritykset  
  6 (0.2949): suunnitella prosessi  

---------------- LANGUAGE = eng -------------------  

TOP-6 similarities of type "occupations" for utterance "prepare, deliver and organize mail":  
  1 (0.2621): mail clerk  
  2 (0.2514): postman/postwoman  
  3 (0.2148): motorcycle delivery person  
  4 (0.1951): car and van delivery driver  
  5 (0.1542): radiation therapist  
  6 (0.1379): construction general contractor  
TOP-6 similarities of type "targeted skills" for utterance "prepare, deliver and organize mail":  
  1 (0.5692): organise mail deliveries  
  2 (0.3324): preparation for child delivery  
  3 (0.3237): handle mail  
  4 (0.3191): work in an organised manner  
  5 (0.2991): oversee delivery of fuel  
  6 (0.2919): register mail  

TOP-6 similarities of type "occupations" for utterance "helping with wheelchair patients":  
  1 (0.1602): palliative care social worker  
  2 (0.1490): hospital social worker  
  3 (0.1339): tyre fitter  
  4 (0.1325): nurse assistant  
  5 (0.1151): dental chairside assistant  
  6 (0.1068): patient transport services driver  
TOP-6 similarities of type "targeted skills" for utterance "helping with wheelchair patients":  
  1 (0.3523): use of special equipment for daily activities  
  2 (0.3176): transfer patients  
  3 (0.2611): assess patients after surgery  
  4 (0.2413): educate patient's relations on care  
  5 (0.2354): plan patient menus  
  6 (0.2099): instruct on the use of special equipment for daily activities  

TOP-6 similarities of type "occupations" for utterance "taking care of small children, feeding, clothing and playing":  
  1 (0.3462): clothing operations manager  
  2 (0.3392): child care coordinator  
  3 (0.3243): child welfare worker  
  4 (0.3178): clothing shop manager  
  5 (0.3095): clothing CAD technician  
  6 (0.3056): child care worker  
TOP-6 similarities of type "targeted skills" for utterance "taking care of small children, feeding, clothing and playing":  
  1 (0.4852): play with children  
  2 (0.4469): supervise children  
  3 (0.4391): caring for children  
  4 (0.4240): implement care programmes for children  
  5 (0.3963): clothing industry  
  6 (0.3899): manufacturing of children clothing  

TOP-6 similarities of type "occupations" for utterance "worked as a waiter in a restaurant, also prepared food and serving it.":  
  1 (0.4567): waiter/waitress  
  2 (0.3966): restaurant host/restaurant hostess  
  3 (0.3911): quick service restaurant team leader  
  4 (0.3663): quick service restaurant crew member  
  5 (0.3380): restaurant manager  
  6 (0.3130):  food service worker  
TOP-6 similarities of type "targeted skills" for utterance "worked as a waiter in a restaurant, also prepared food and serving it.":  
  1 (0.5384): prepare the restaurant for service  
  2 (0.2735): welcome restaurant guests  
  3 (0.1633): serve food in table service  
  4 (0.1549): process reservations  
  5 (0.1441): ensure food quality  
  6 (0.1415): supervise food quality  

```

16.4.2022:
Added a FinBERT model trained with PARLAI module. In order to use this, install PARLAI and download the agent pickle from the following link: https://1drv.ms/u/s!Ai_vcSNOnJbkiFdszi4aSmQmKDkE?e=B1wA5T
This is somewhat better than TF-IDF (as one might expect). Still not great and we need to rely on those manually adjusted thresholds to guide recommendations.

I also modified the test scripts so that I can process longer texts. Idea is that the text is split in sentences which are processed individually. Occupations are applied to limit the search over all sentences.

