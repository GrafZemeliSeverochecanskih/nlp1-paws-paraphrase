# P2 error examples

Out-of-fold predictions, seed 0. Systems: bilstm+siamese, lstm+siamese.

## all 2 P2 systems wrong (3350 of 20000 pairs)

- id 1050, gold **paraphrase**, same_word_set 0, order_change 0.55, oov_rate 0.00
  - S1: During the American Civil War , Indiana was the site of the Battle of Corydon , the only official pitched battle waged in Corydon .
  - S2: Indiana was the site of the Battle of Corydon during the American Civil War , the only official battle in Corydon .
- id 6536, gold **paraphrase**, same_word_set 0, order_change 0.06, oov_rate 0.00
  - S1: On March 17 , 2015 , Actavis acquired Allergan and took the Allergan name .
  - S2: On 17 March 2015 , Actavis acquired Allergan and adopted the name Allergan .
- id 7566, gold **different**, same_word_set 0, order_change 0.14, oov_rate 0.05
  - S1: It is hosted by Guy Zu-Aretz and Eli Mizrahi with judges Yarden Harel and the latest Uri Paster and Michal Amdurski .
  - S2: It is hosted by Guy Zu-Aretz and Yarden Harel with judges Eli Mizrahi and the Newest Uri Paster and Michal Amdurski .
- id 9570, gold **different**, same_word_set 0, order_change 0.17, oov_rate 0.00
  - S1: In case of many problems it is more convenient to work with D and the free charges than with E and the total charge .
  - S2: In the case of many problems , it is more convenient to work with D and total fees than with E and the free charge .
- id 10110, gold **paraphrase**, same_word_set 0, order_change 0.69, oov_rate 0.00
  - S1: On 31 August 1819 , Wollstonecraft met in Sydney on board the ship `` Grenada '' .
  - S2: Wollstonecraft arrived in Sydney on board the ship `` Grenada '' on 31 August 1819 .
- id 10498, gold **paraphrase**, same_word_set 0, order_change 0.09, oov_rate 0.08
  - S1: Drietoma is a village and municipality in the Trenčín District in the Trenčín Region of northwestern Slovakia .
  - S2: Drietoma is a village and municipality in the region of Trenčín in the district Trenčín in north-western Slovakia .
- id 11964, gold **paraphrase**, same_word_set 0, order_change 0.26, oov_rate 0.00
  - S1: A few years later , General Obasanjo was released and pardoned after Abacha died and after General Abdulsalami Abubakar took over power .
  - S2: General Obasanjo was released and pardoned a number of years later after Abacha died and after General Abdulsalami Abubakar took power .
- id 12461, gold **different**, same_word_set 0, order_change 0.10, oov_rate 0.00
  - S1: He continued in this post when Winston Churchill came to power in 1940 and was also Churchill 's general postmaster between 1943 and 1945 .
  - S2: He continued in this post also when Winston Churchill came to power in 1940 , and was then Postmaster General under Churchill between 1943 and 1945 .
- id 18450, gold **different**, same_word_set 0, order_change 0.00, oov_rate 0.00
  - S1: After calling Tinkerer who calls him an updated version of his Clash suit , Clayton makes up Mendel Stromm .
  - S2: After calling Tinkerer who calls him an updated version of his clash suit , Clayton Mendel Stromm produces .
- id 19097, gold **different**, same_word_set 1, order_change 0.21, oov_rate 0.00
  - S1: It is rare in its unhybridized form .
  - S2: It is in its rare form unhybridized .
