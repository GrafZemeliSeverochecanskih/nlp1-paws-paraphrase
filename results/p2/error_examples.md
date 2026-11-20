# P2 error examples

Out-of-fold predictions, seed 0. Systems: bilstm+cross, bilstm+siamese+h256, bilstm+siamese+h64, bilstm+siamese, lstm+cross, lstm+siamese+trainable, lstm+siamese, pool+cross+sum, pool+cross, pool+siamese+sum, pool+siamese, glove6B-50d:bilstm+siamese, glove6B-50d:pool+siamese, glove6B-100d:bilstm+siamese, glove6B-100d:pool+siamese, glove6B-300d:bilstm+siamese, glove6B-300d:pool+siamese.

## lstm+siamese right, pool+cross wrong (5893 of 20000 pairs)

- id 4684, gold **paraphrase**, same_word_set 0, order_change 0.19, oov_rate 0.00
  - S1: 1946 -- The state of Prussia becomes abolished after the Second World War . Meppen is part of the newly created `` Land '' of Lower Saxony .
  - S2: 1946 -- The state of Prussia is abolished after the Second World War , Meppen becomes part of the newly created `` Land '' Lower Saxony .
- id 7661, gold **paraphrase**, same_word_set 0, order_change 0.00, oov_rate 0.00
  - S1: In 1976 , a version of Wonder Girl named Wonder Woman appeared in the Drusilla television series and was played by Debra Winger .
  - S2: In 1976 , a version of Wonder Girl called Wonder Woman appeared in the Drusilla TV series and was played by Debra Winger .
- id 7732, gold **paraphrase**, same_word_set 0, order_change 0.10, oov_rate 0.00
  - S1: The species was discovered in 2013 and named and described in 2015 by Samuel P. Iglésias and Lou Frotté .
  - S2: The species were discovered in 2013 and were named and described by Samuel P. Iglésias and Lou Frotté in 2015 .
- id 9409, gold **different**, same_word_set 0, order_change 0.51, oov_rate 0.00
  - S1: In many provinces , mosques were bombed and Hui was slaughtered or destroyed by Japanese troops .
  - S2: Mosques were destroyed and in many provinces Hui were slaughtered by Japanese troops or bombed .
- id 13032, gold **paraphrase**, same_word_set 0, order_change 0.13, oov_rate 0.04
  - S1: He has a son ( born 2000 ) from a former relationship with TV presenter Angela Gessmann , who married Birgit Schrowange in July 2011 .
  - S2: He has a son ( born 2000 ) from a previous relationship with TV presenter Angela Gessmann . In July 2011 Lanz married Birgit Schrowange .
- id 15239, gold **paraphrase**, same_word_set 0, order_change 0.07, oov_rate 0.08
  - S1: Because of the high cost of Stromectol , the lifelong formula Ivomec can be used government programs are needed to help citizens finance veterinary medicines .
  - S2: Because of the high cost of Stromectol , the lifelong formula Ivomec can be used . Government programs are needed to help citizens finance veterinary medication .
- id 15649, gold **paraphrase**, same_word_set 1, order_change 0.27, oov_rate 0.00
  - S1: In the evening , finally a bowl came with a small soup with a thin piece of bread .
  - S2: Finally , in the evening , came a bowl of small soup with a thin piece of bread .
- id 15947, gold **different**, same_word_set 0, order_change 0.20, oov_rate 0.00
  - S1: In most countries , a different organisation , a scout group , combines local sections into a single body .
  - S2: In most countries a local organisation , a Scout Group , combines different sections together into a single body .
- id 19267, gold **paraphrase**, same_word_set 0, order_change 0.12, oov_rate 0.00
  - S1: He said that the ordinary Cairenes believed that there was an American conspiracy to attack EgyptAir 990 and that the Americans were covering up the fact .
  - S2: He said that ordinary Cairenes believed that there was an American conspiracy to attack EgyptAir 990 , and that the Americans covered up the fact .
- id 19603, gold **paraphrase**, same_word_set 0, order_change 0.28, oov_rate 0.00
  - S1: A change in the team that played the second test was made Pakistan : Alimuddin replaced Intikhab Alam .
  - S2: Pakistan made one change in the team that played the Second Test : Alimuddin replaced Intikhab Alam .

## all 17 P2 systems wrong (560 of 20000 pairs)

- id 1403, gold **paraphrase**, same_word_set 0, order_change 0.14, oov_rate 0.00
  - S1: It is located at 142 South Rexford Drive in Beverly Hills , California . It stands opposite the First Church of Christ , Scientist , Beverly Hills .
  - S2: It is located at 142 South Rexford Drive in Beverly Hills . It is opposite the First Church of Christ , scientist , Beverly Hills , California .
- id 9365, gold **paraphrase**, same_word_set 0, order_change 0.00, oov_rate 0.17
  - S1: The Limbăşelu River is a tributary of the Cenușeroaia River in Romania .
  - S2: The river LimbÄ Å elu is a tributary of the River Cenué eroaia in Romania .
- id 10984, gold **paraphrase**, same_word_set 0, order_change 0.95, oov_rate 0.07
  - S1: After being demoted to Triple-A Las Vegas , LaRoche was recalled on September 1 .
  - S2: LaRoche was recalled on 1 September after being degraded to Triple-A Las Vegas .
- id 13122, gold **paraphrase**, same_word_set 1, order_change 0.33, oov_rate 0.00
  - S1: The joint management of the memorial by the United States Navy and the National Park Service was founded on 9 September 1980 .
  - S2: The joint management of the memorial was founded by the National Park Service and the United States Navy on 9 September 1980 .
- id 13887, gold **paraphrase**, same_word_set 0, order_change 0.15, oov_rate 0.00
  - S1: Hucknall Town was a railway station on the Great Northern Railway 's Shirebrook to Nottingham line .
  - S2: Hucknall Town was a railway station on the Shirebrook to Nottingham - line of the Great Northern Railway .
- id 14418, gold **paraphrase**, same_word_set 0, order_change 0.00, oov_rate 0.00
  - S1: The underlying typology ( according to the nature of the second amount ) is determined by the needs of users of the variance information and may include e.g . :
  - S2: The underlying typology ( according to the type of second amount ) is determined by the needs of the users of the variance information and may z .
- id 15260, gold **paraphrase**, same_word_set 0, order_change 0.82, oov_rate 0.00
  - S1: Srinivas approached Rahman , who was encouraged by the use of fresh voices in `` Roja '' .
  - S2: Encouraged by the use of fresh voices in `` Roja '' , Srinivas approached Rahman .
- id 17413, gold **paraphrase**, same_word_set 0, order_change 0.00, oov_rate 0.15
  - S1: The Coruia River is a tributary of the river LÄ puÅ in Romania .
  - S2: The Coruia River is a tributary of the Lăpuş River in Romania .
- id 17642, gold **paraphrase**, same_word_set 0, order_change 0.26, oov_rate 0.00
  - S1: Mimi Sodré was a student at the Naval Academy when he was interested in scouting after reading a book by Baden Powell called Scouting for Boys .
  - S2: Mimi Sodré was a student at the Naval Academy when after reading a book by Baden Powell named Scouting for Boys , he became interested in Scouting .
- id 18688, gold **paraphrase**, same_word_set 0, order_change 0.82, oov_rate 0.00
  - S1: In addition to the German teaching programme , the official Mexican language was taught only as a foreign language .
  - S2: The official Mexican language was only taught as a foreign language in addition to the German teaching program .
