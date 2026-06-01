# KnowEdit ZsRE LAwF Benchmark

This diagnostic adapts KnowEdit/ZsRE QA edits to sparse token-level correction.
It is a real-data external-validity check, not a full model-editing benchmark.

- Model: `/root/lawf_experiment/modelscope_cache/Qwen/Qwen3___5-9B`
- Data source: `https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/ZsRE/ZsRE-test-all.json`
- Selected edits: `128` / raw `1301`
- Anchor tokens: `128` / `476` (26.89%)
- Anchor policy: `first_token`
- Anchor target probability: `0.9`
- Anchor probability tolerance: `0.0`
- Steps: `16`
- LoRA: r=`8`, alpha=`16`
- LAwF: alpha=`1.0`, beta=`1.0`, normalization=`token_mean`

## Summary

| Model | Direct CE | Rephrase CE | Portability CE | Locality KL | Retention KL | Train non-anchor KL | Train anchor CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 5.499536 | 5.414731 | 5.336004 | 0.000000 | 0.000000 | - | - |
| sft | 0.003412 | 0.148155 | 3.617231 | 7.546691 | 1.292836 | 11.137316 | 0.003716 |
| sft_kl | 0.497652 | 0.678771 | 2.897303 | 2.009527 | 0.250492 | 0.365642 | 0.082409 |

## Edit Set

| ID | Subject | Target | Old answer | Locality probes | Portability probes |
| --- | --- | --- | --- | ---: | ---: |
| zsre_0 | Epaspidoceras | Noctuidae | Aspidoceratidae | 2 | 1 |
| zsre_1 | ZIC3 | male | human | 2 | 1 |
| zsre_2 | Louise Grandjean | mezzo soprano | soprano | 2 | 1 |
| zsre_3 | Wang Jipeng | Wang Chonghua | Wang Yanjun | 2 | 1 |
| zsre_4 | Charlotte of Schaumburg-Lippe | Charlotte of Bourbon-Parma | Princess Bathildis of Anhalt-Dessau | 2 | 1 |
| zsre_5 | Butterfly Cluster | Orion | Scorpius | 2 | 1 |
| zsre_6 | Juan María Bordaberry | Gabrielle Bordaberry | Domingo Bordaberry | 2 | 1 |
| zsre_7 | Javan surili | critically threatened | endangered species | 2 | 1 |
| zsre_9 | Runaway Sunday | Motown | Virgin Records | 2 | 1 |
| zsre_10 | Southern California Fusion | USL First Division | National Premier Soccer League | 2 | 1 |
| zsre_12 | Joseph Papp | pneumonia | prostate cancer | 2 | 1 |
| zsre_13 | Holmenkollen Chapel | Norwegian Institute of Technology | Holger Sinding-Larsen | 2 | 1 |
| zsre_14 | Marc Moulin | Catherine Moulin | Jeanine Moulin | 2 | 1 |
| zsre_15 | Nicolas Raffault | Arizona Coyotes | Lyon OU | 2 | 1 |
| zsre_16 | Charity Creek | Charity River | Parramatta River | 2 | 1 |
| zsre_17 | Nils Palme | Lau Lauritzen | Sven Palme | 2 | 1 |
| zsre_18 | Bali myna | myna | critically endangered | 2 | 1 |
| zsre_19 | Coevorden | Alexander Coevorden | cow | 2 | 1 |
| zsre_20 | Pedro Magallanes | Colombia | Argentina | 2 | 1 |
| zsre_21 | Heroes Chronicles | Chris Riddell | Jon Van Caneghem | 2 | 1 |
| zsre_22 | Archduchess Mechthildis of Austria | Infanta Maria Theresa of Portugal | Archduke Charles Stephen of Austria | 2 | 1 |
| zsre_23 | Ang TV | Sri Lanka | Philippines | 2 | 1 |
| zsre_24 | Alexanderson alternator | Ernest Alexanderson | Ernst Alexanderson | 2 | 1 |
| zsre_25 | Mallory Reaves | Lalli Reaves | Brynne Chandler | 2 | 1 |
| zsre_26 | Harlo Jones | pneumonia | stroke | 2 | 1 |
| zsre_28 | Thomas the Tank Engine | William Orpen | Wilbert Awdry | 2 | 1 |
| zsre_29 | Alec Rose | Spanish Civil War | World War II | 2 | 1 |
| zsre_30 | The Smothers Brothers Comedy Hour | NBC | CBS | 2 | 1 |
| zsre_31 | Anthony Delon | Alma Delon | Nathalie Delon | 2 | 1 |
| zsre_32 | Petteri Nummelin | Columbus Blue Bombers | Minnesota Wild | 2 | 1 |
| zsre_33 | CXCL10 | male | human | 2 | 1 |
| zsre_34 | Miliolacea | Agaricaceae | Miliolida | 2 | 1 |
| zsre_35 | Andy Luckey | Luckey the Dolphin | Bud Luckey | 2 | 1 |
| zsre_36 | Prince Karl Johann of Liechtenstein | Princess Sophie of Greece and Denmark | Landgravine Josepha of Fürstenberg-Weitra | 2 | 1 |
| zsre_37 | JS 7.62 | 1961 | 2005 | 2 | 1 |
| zsre_38 | Air France Flight 447 | 12 July 1944 | 1 June 2009 | 2 | 1 |
| zsre_39 | Dexter Coffin Bridge | Connecticut Creek | Connecticut River | 2 | 1 |
| zsre_40 | Ridgely Gaither | World War II | Korean War | 2 | 1 |
| zsre_41 | Jon Skolmen | Linda Skolmen | Eli Skolmen Ryg | 2 | 1 |
| zsre_42 | Sigil Games Online | 1999 | 2002 | 2 | 1 |
| zsre_43 | Ralph Habib | Chicago | Paris | 2 | 1 |
| zsre_44 | Pleine Vie | Coptic | French | 2 | 1 |
| zsre_45 | USA-126 | 26 September126 | 16 July 1996 | 2 | 1 |
| zsre_46 | Eteocles | Dagobert | Oedipus | 2 | 1 |
| zsre_47 | Tristan von Lahnstein | Inspector Morse | Verbotene Liebe | 2 | 1 |
| zsre_48 | Street Rod 2 | Sierra Entertainment | California Dreams | 2 | 1 |
| zsre_50 | Suggan Buggan River | Bass Strait | Snowy River | 2 | 1 |
| zsre_51 | Arwen | Doris | Celebrían | 2 | 1 |
| zsre_52 | Hans Ulrik Gyldenløve | Marie Louise Föhse | Karen Andersdatter | 2 | 1 |
| zsre_53 | NGC 5985 | Boötes | Draco | 2 | 1 |
| zsre_54 | Tau Herculis | Hornax | Hercules | 2 | 1 |
| zsre_55 | When China Met Africa | Famous Players Television | Speakit Films | 2 | 1 |
| zsre_56 | 503 Evelyn | 17 503 | 19 January 1903 | 2 | 1 |
| zsre_57 | JS 7.62 | 1963 | 2005 | 2 | 1 |
| zsre_58 | Svend Poulsen | War of 1812 | Torstenson War | 2 | 1 |
| zsre_59 | Juno Temple | Jupiter | Julien Temple | 2 | 1 |
| zsre_61 | Alien Front Online | 2K Games | Sega | 2 | 1 |
| zsre_62 | HD 177808 | Lynx | Lyra | 2 | 1 |
| zsre_63 | Alec Rose | Spanish Civil War | World War II | 2 | 1 |
| zsre_64 | HD 85622 | Carina | Vela | 2 | 1 |
| zsre_65 | Holmenkollen Chapel | Inigo Jones | Holger Sinding-Larsen | 2 | 1 |
| zsre_66 | Old Quebec Street Mall | 2002 | 2003 | 2 | 1 |
| zsre_67 | GNOME Chess | Python | Vala | 2 | 1 |
| zsre_68 | Hannelore Kohl | John Kohl | Helmut Kohl | 2 | 1 |
| zsre_69 | Kishar | Bhutan | Lahamu | 2 | 1 |
| zsre_70 | Rhinocoryne | Noctuidae | Batillariidae | 2 | 1 |
| zsre_71 | Alexander Aris | Irving Kane Pond | Aung San Suu Kyi | 2 | 1 |
| zsre_72 | Nermin Čeliković | 8 September 1981 | 27 November 1980 | 2 | 1 |
| zsre_73 | USA-64 | 3 December 1992 | 1 October 1990 | 2 | 1 |
| zsre_74 | Herbert T. Levack | American Civil War | World War II | 2 | 1 |
| zsre_75 | Leonor, Princess of Asturias | Leonor III of Spain | Felipe VI | 2 | 1 |
| zsre_76 | Bethune Memorial House | Mary Bethune | Norman Bethune | 2 | 1 |
| zsre_77 | Tupolev | Kazan Airlines | United Aircraft Corporation | 2 | 1 |
| zsre_78 | Vindhya Pradesh | 1856 | 1956 | 2 | 1 |
| zsre_79 | Zdeněk Nejedlý | Slovakia | Czech Republic | 2 | 1 |
| zsre_80 | Gabb's snail | Lymantriurus | Micrarionta | 2 | 1 |
| zsre_81 | HD 180902 | Ophiuchus | Sagittarius | 2 | 1 |
| zsre_82 | Chitinase | male | human | 2 | 1 |
| zsre_84 | Odelay | Academy Award for Best Picture | Grammy Award for Album of the Year | 2 | 1 |
| zsre_85 | MSH3 | male | human | 2 | 1 |
| zsre_87 | Gwendolyn Killebrew | mezzo soprano | contralto | 2 | 1 |
| zsre_88 | Gate Dancer | Dancer of the East | Sovereign Dancer | 2 | 1 |
| zsre_89 | Noticias ECO | publishing | news | 2 | 1 |
| zsre_90 | Order of the Black Eagle | 1915 | 1918 | 2 | 1 |
| zsre_91 | Atreus | Darius III | Aerope | 2 | 1 |
| zsre_93 | Bioscience Horizons | Wiley-Blackwell | Oxford University Press | 2 | 1 |
| zsre_94 | HD 125658 | Leo Minor | Boötes | 2 | 1 |
| zsre_95 | Colorhythm | Lil' Mo | Hitomi Yaida | 2 | 1 |
| zsre_96 | Maria Antonia of Austria | Elisabeth of Bavaria | Margaret Theresa of Spain | 2 | 1 |
| zsre_97 | Estate Exchange | Welton Becket | Thomas Worthington | 2 | 1 |
| zsre_98 | Château Mont-Royal | Édouard Niermans | Guillaume Tronchet | 2 | 1 |
| zsre_99 | Moses Magnum | Noon Universe | Marvel Universe | 2 | 1 |
| zsre_100 | Fritz X | 1940 | 1943 | 2 | 1 |
| zsre_101 | Mongenast Ministry | 1941 | 6 November 1915 | 2 | 1 |
| zsre_102 | Halenia | Geometridae | Gentianaceae | 2 | 1 |
| zsre_103 | Ariadne musica | orchestra | organ | 2 | 1 |
| zsre_104 | Simple Souls | TSR | Pathé Exchange | 2 | 1 |
| zsre_105 | Anhui musk deer | vulnerable | endangered species | 2 | 1 |
| zsre_106 | Andrew Toney | Vancouver Canucks | Philadelphia 76ers | 2 | 1 |
| zsre_107 | Horkos | Amenhotep III | Eris | 2 | 1 |
| zsre_108 | The Last Days | Peter Bogdanovich | James Moll | 2 | 1 |
| zsre_109 | Fimpen | Wolfgang Becker | Bo Widerberg | 2 | 1 |
| zsre_110 | JS 7.62 | 1966 | 2005 | 2 | 1 |
| zsre_111 | Philipp Orter | 20 April 1894 | 16 February 1994 | 2 | 1 |
| zsre_112 | Anguispira | Crambidae | Discidae | 2 | 1 |
| zsre_113 | Melissa Magstadt | member of the Illinois House of Representatives | member of the South Dakota House of Representatives | 2 | 1 |
| zsre_114 | MAT-49 | 2011 | 1949 | 2 | 1 |
| zsre_115 | Fritz X | 1940 | 1943 | 2 | 1 |
| zsre_116 | Josef Lada | Seville | Prague | 2 | 1 |
| zsre_117 | Fambly 42 | Warner Bros | Recess Records | 2 | 1 |
| zsre_118 | José Wilker | yellow fever | heart attack | 2 | 1 |
| zsre_119 | Southern Crab Nebula | Cygnus | Centaurus | 2 | 1 |
| zsre_120 | Esther Bloom | The Divine Comedy | Hollyoaks | 2 | 1 |
| zsre_121 | British Rail Class 47 | Trimark | Brush Traction | 2 | 1 |
| zsre_122 | Nikolaos Kriezotis | Greco-Italian War | Greek War of Independence | 2 | 1 |
| zsre_123 | The Queen and the Dreams | Motown | Taihe Rye Music | 2 | 1 |
| zsre_124 | Antônio Cardoso | Rio Grande do Sul | Bahia | 2 | 1 |
| zsre_125 | Lloyd Thomas | Spanish Civil War | World War II | 2 | 1 |
| zsre_126 | Angolan African dormouse | Lecithoceridae | Graphiurus | 2 | 1 |
| zsre_127 | Lloyd Thomas | Spanish Civil War | World War II | 2 | 1 |
| zsre_128 | Leucyl/cystinyl aminopeptidase | male | human | 2 | 1 |
| zsre_129 | Cercestis | Noctuidae | Araceae | 2 | 1 |
| zsre_130 | NGC 2 | Dorado | Pegasus | 2 | 1 |
| zsre_131 | Wolf 359 | Auriga | Leo | 2 | 1 |
| zsre_132 | Mohammad Naseem | Tajikistan | Birmingham | 2 | 1 |
| zsre_133 | United Abominations | Arista Records | Roadrunner Records | 2 | 1 |
| zsre_134 | Beatriz Balzi | 17 May 2015 | 2001 | 2 | 1 |
| zsre_135 | Prince Emmanuel of Belgium | Prince Philippe of Belgium | Philippe of Belgium | 2 | 1 |