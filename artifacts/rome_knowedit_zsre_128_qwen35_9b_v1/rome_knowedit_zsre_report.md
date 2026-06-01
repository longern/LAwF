# ROME KnowEdit-ZsRE Baseline

This baseline applies one ROME edit at a time on the adapted KnowEdit-ZsRE short-answer benchmark.
Each edit is evaluated and then reverted before the next edit.

- Model: `/root/lawf_experiment/modelscope_cache/Qwen/Qwen3___5-9B`
- Selected edits: `128`
- ROME layer: `[6]`
- ROME v-loss layer: `31`

## Summary

| Setting | Direct CE | Rephrase CE | Portability CE | Locality KL | Retention KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 5.499536 | 5.414731 | 5.336004 | 0.000000 | 0.000000 |
| ROME | 2.602192 | 2.688106 | 4.835483 | 1.093000 | 0.001445 |

## Per-Edit Results

| ID | Subject | Target | Direct CE Before | Direct CE After | Rephrase CE After | Portability CE After | Locality KL | Retention KL |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zsre_0 | Epaspidoceras | Noctuidae | 3.803 | 0.271 | 0.281 | 3.828 | 1.8966 | 0.0012 |
| zsre_1 | ZIC3 | male | 13.363 | 7.898 | 7.483 | 9.960 | 0.6093 | 0.0037 |
| zsre_2 | Louise Grandjean | mezzo soprano | 6.848 | 4.941 | 5.235 | 4.563 | 0.2589 | 0.0017 |
| zsre_3 | Wang Jipeng | Wang Chonghua | 3.518 | 0.776 | 0.767 | 4.305 | 0.3769 | 0.0014 |
| zsre_4 | Charlotte of Schaumburg-Lippe | Charlotte of Bourbon-Parma | 3.255 | 0.945 | 0.963 | 4.816 | 0.8110 | 0.0006 |
| zsre_5 | Butterfly Cluster | Orion | 4.263 | 4.442 | 5.109 | 4.178 | 1.0949 | 0.0011 |
| zsre_6 | Juan María Bordaberry | Gabrielle Bordaberry | 4.140 | 0.682 | 1.068 | 5.332 | 1.4142 | 0.0009 |
| zsre_7 | Javan surili | critically threatened | 7.144 | 2.687 | 4.375 | 2.834 | 0.2087 | 0.0006 |
| zsre_9 | Runaway Sunday | Motown | 5.820 | 1.561 | 1.330 | 6.653 | 1.4003 | 0.0030 |
| zsre_10 | Southern California Fusion | USL First Division | 5.133 | 2.460 | 2.111 | 6.096 | 0.5122 | 0.0005 |
| zsre_12 | Joseph Papp | pneumonia | 4.145 | 5.148 | 4.284 | 5.995 | 0.5370 | 0.0015 |
| zsre_13 | Holmenkollen Chapel | Norwegian Institute of Technology | 5.181 | 1.249 | 1.909 | 5.686 | 0.9519 | 0.0010 |
| zsre_14 | Marc Moulin | Catherine Moulin | 3.379 | 2.499 | 2.127 | 5.797 | 0.2059 | 0.0015 |
| zsre_15 | Nicolas Raffault | Arizona Coyotes | 5.670 | 1.796 | 1.792 | 3.677 | 0.8535 | 0.0019 |
| zsre_16 | Charity Creek | Charity River | 4.318 | 1.930 | 2.031 | 6.248 | 0.2276 | 0.0009 |
| zsre_17 | Nils Palme | Lau Lauritzen | 5.189 | 2.968 | 2.955 | 9.363 | 0.2144 | 0.0006 |
| zsre_18 | Bali myna | myna | 9.452 | 5.688 | 3.311 | 11.685 | 0.3825 | 0.0007 |
| zsre_19 | Coevorden | Alexander Coevorden | 4.643 | 1.870 | 1.637 | 4.908 | 0.8275 | 0.0015 |
| zsre_20 | Pedro Magallanes | Colombia | 3.121 | 0.819 | 1.133 | 3.098 | 1.2817 | 0.0004 |
| zsre_21 | Heroes Chronicles | Chris Riddell | 4.878 | 2.410 | 2.516 | 5.103 | 0.5883 | 0.0014 |
| zsre_22 | Archduchess Mechthildis of Austria | Infanta Maria Theresa of Portugal | 4.764 | 2.680 | 3.912 | 4.314 | 0.2334 | 0.0008 |
| zsre_23 | Ang TV | Sri Lanka | 3.949 | 2.407 | 2.598 | 3.023 | 0.7778 | 0.0061 |
| zsre_24 | Alexanderson alternator | Ernest Alexanderson | 5.711 | 4.363 | 4.610 | 5.669 | 0.4126 | 0.0006 |
| zsre_25 | Mallory Reaves | Lalli Reaves | 4.640 | 1.660 | 1.590 | 4.201 | 1.0077 | 0.0020 |
| zsre_26 | Harlo Jones | pneumonia | 3.898 | 3.290 | 5.869 | 5.377 | 0.8849 | 0.0015 |
| zsre_28 | Thomas the Tank Engine | William Orpen | 7.244 | 2.549 | 6.279 | 2.595 | 3.1947 | 0.0015 |
| zsre_29 | Alec Rose | Spanish Civil War | 4.656 | 1.053 | 0.805 | 1.872 | 2.5160 | 0.0023 |
| zsre_30 | The Smothers Brothers Comedy Hour | NBC | 8.123 | 4.076 | 2.695 | 4.644 | 0.4951 | 0.0009 |
| zsre_31 | Anthony Delon | Alma Delon | 3.830 | 2.826 | 3.433 | 5.021 | 0.6458 | 0.0013 |
| zsre_32 | Petteri Nummelin | Columbus Blue Bombers | 4.723 | 2.532 | 2.283 | 6.135 | 0.2315 | 0.0014 |
| zsre_33 | CXCL10 | male | 11.986 | 7.311 | 5.711 | 9.333 | 1.1160 | 0.0013 |
| zsre_34 | Miliolacea | Agaricaceae | 3.529 | 0.972 | 0.868 | 1.181 | 0.3048 | 0.0005 |
| zsre_35 | Andy Luckey | Luckey the Dolphin | 6.215 | 1.687 | 1.756 | 7.524 | 1.3694 | 0.0014 |
| zsre_36 | Prince Karl Johann of Liechtenstein | Princess Sophie of Greece and Denmark | 3.237 | 2.279 | 2.099 | 4.787 | 0.3206 | 0.0008 |
| zsre_37 | JS 7.62 | 1961 | 3.270 | 1.442 | 1.814 | 4.643 | 0.1416 | 0.0011 |
| zsre_38 | Air France Flight 447 | 12 July 1944 | 3.471 | 0.950 | 1.223 | 2.828 | 1.0451 | 0.0005 |
| zsre_39 | Dexter Coffin Bridge | Connecticut Creek | 6.675 | 1.454 | 1.488 | 4.401 | 1.9905 | 0.0009 |
| zsre_40 | Ridgely Gaither | World War II | 3.816 | 0.767 | 0.883 | 3.273 | 1.0992 | 0.0017 |
| zsre_41 | Jon Skolmen | Linda Skolmen | 3.363 | 2.371 | 2.043 | 3.600 | 0.9241 | 0.0013 |
| zsre_42 | Sigil Games Online | 1999 | 3.001 | 2.633 | 2.847 | 4.054 | 1.3099 | 0.0008 |
| zsre_43 | Ralph Habib | Chicago | 7.957 | 1.681 | 1.130 | 2.774 | 0.2034 | 0.0012 |
| zsre_44 | Pleine Vie | Coptic | 6.205 | 1.567 | 1.205 | 2.891 | 0.5745 | 0.0009 |
| zsre_45 | USA-126 | 26 September126 | 5.339 | 0.911 | 0.797 | 6.051 | 0.4634 | 0.0026 |
| zsre_46 | Eteocles | Dagobert | 6.241 | 2.407 | 2.467 | 5.189 | 1.0004 | 0.0018 |
| zsre_47 | Tristan von Lahnstein | Inspector Morse | 9.936 | 2.002 | 2.812 | 4.599 | 0.6616 | 0.0014 |
| zsre_48 | Street Rod 2 | Sierra Entertainment | 6.204 | 5.802 | 5.371 | 7.073 | 0.4974 | 0.0011 |
| zsre_50 | Suggan Buggan River | Bass Strait | 4.522 | 1.511 | 1.707 | 5.085 | 0.5310 | 0.0009 |
| zsre_51 | Arwen | Doris | 9.337 | 6.448 | 6.801 | 9.949 | 5.2716 | 0.0011 |
| zsre_52 | Hans Ulrik Gyldenløve | Marie Louise Föhse | 6.649 | 3.007 | 2.879 | 5.329 | 0.9991 | 0.0011 |
| zsre_53 | NGC 5985 | Boötes | 4.807 | 2.178 | 2.237 | 5.967 | 1.0880 | 0.0008 |
| zsre_54 | Tau Herculis | Hornax | 11.555 | 3.920 | 3.915 | 5.521 | 3.5992 | 0.0007 |
| zsre_55 | When China Met Africa | Famous Players Television | 7.890 | 2.508 | 3.035 | 7.282 | 1.5318 | 0.0010 |
| zsre_56 | 503 Evelyn | 17 503 | 5.088 | 2.379 | 1.588 | 5.163 | 0.6702 | 0.0022 |
| zsre_57 | JS 7.62 | 1963 | 3.118 | 1.756 | 1.542 | 3.440 | 0.2753 | 0.0010 |
| zsre_58 | Svend Poulsen | War of 1812 | 2.380 | 0.361 | 0.414 | 2.499 | 1.2931 | 0.0014 |
| zsre_59 | Juno Temple | Jupiter | 6.463 | 3.745 | 3.599 | 10.513 | 1.7706 | 0.0029 |
| zsre_61 | Alien Front Online | 2K Games | 4.545 | 3.038 | 2.927 | 3.381 | 0.3481 | 0.0010 |
| zsre_62 | HD 177808 | Lynx | 4.978 | 2.612 | 1.097 | 2.577 | 0.7790 | 0.0010 |
| zsre_63 | Alec Rose | Spanish Civil War | 4.305 | 0.302 | 0.271 | 1.752 | 1.6399 | 0.0013 |
| zsre_64 | HD 85622 | Carina | 5.553 | 1.354 | 1.200 | 5.523 | 0.6117 | 0.0009 |
| zsre_65 | Holmenkollen Chapel | Inigo Jones | 4.943 | 3.447 | 4.372 | 4.898 | 0.7498 | 0.0007 |
| zsre_66 | Old Quebec Street Mall | 2002 | 4.040 | 2.763 | 2.611 | 3.865 | 1.0000 | 0.0012 |
| zsre_67 | GNOME Chess | Python | 7.077 | 3.005 | 2.505 | 3.143 | 0.3865 | 0.0009 |
| zsre_68 | Hannelore Kohl | John Kohl | 6.814 | 3.960 | 4.012 | 6.836 | 5.3314 | 0.0018 |
| zsre_69 | Kishar | Bhutan | 5.297 | 1.808 | 2.818 | 5.107 | 3.6484 | 0.0018 |
| zsre_70 | Rhinocoryne | Noctuidae | 3.143 | 1.950 | 1.819 | 2.929 | 0.4234 | 0.0033 |
| zsre_71 | Alexander Aris | Irving Kane Pond | 7.071 | 2.264 | 2.315 | 8.535 | 0.9757 | 0.0017 |
| zsre_72 | Nermin Čeliković | 8 September 1981 | 2.749 | 2.071 | 2.050 | 6.344 | 0.7335 | 0.0006 |
| zsre_73 | USA-64 | 3 December 1992 | 3.125 | 1.114 | 1.079 | 3.091 | 0.8673 | 0.0026 |
| zsre_74 | Herbert T. Levack | American Civil War | 5.072 | 3.076 | 2.754 | 4.373 | 0.0612 | 0.0011 |
| zsre_75 | Leonor, Princess of Asturias | Leonor III of Spain | 5.381 | 2.549 | 2.524 | 5.044 | 0.2747 | 0.0007 |
| zsre_76 | Bethune Memorial House | Mary Bethune | 4.981 | 3.814 | 2.189 | 4.124 | 1.8000 | 0.0009 |
| zsre_77 | Tupolev | Kazan Airlines | 7.279 | 2.739 | 3.227 | 4.345 | 0.5761 | 0.0023 |
| zsre_78 | Vindhya Pradesh | 1856 | 4.165 | 1.860 | 1.184 | 4.539 | 1.1712 | 0.0010 |
| zsre_79 | Zdeněk Nejedlý | Slovakia | 3.315 | 1.439 | 1.533 | 3.002 | 0.4793 | 0.0020 |
| zsre_80 | Gabb's snail | Lymantriurus | 10.785 | 3.069 | 2.930 | 6.264 | 0.6631 | 0.0008 |
| zsre_81 | HD 180902 | Ophiuchus | 4.030 | 2.490 | 3.738 | 4.437 | 1.3718 | 0.0008 |
| zsre_82 | Chitinase | male | 14.152 | 8.678 | 9.027 | 11.448 | 0.8889 | 0.0009 |
| zsre_84 | Odelay | Academy Award for Best Picture | 3.090 | 1.103 | 1.002 | 1.636 | 2.0298 | 0.0020 |
| zsre_85 | MSH3 | male | 12.884 | 5.901 | 5.997 | 8.585 | 0.7585 | 0.0024 |
| zsre_87 | Gwendolyn Killebrew | mezzo soprano | 6.609 | 4.144 | 4.005 | 6.505 | 0.0715 | 0.0014 |
| zsre_88 | Gate Dancer | Dancer of the East | 4.488 | 3.064 | 2.484 | 5.788 | 1.5317 | 0.0017 |
| zsre_89 | Noticias ECO | publishing | 8.160 | 3.345 | 4.169 | 3.522 | 0.9187 | 0.0012 |
| zsre_90 | Order of the Black Eagle | 1915 | 3.930 | 0.655 | 1.037 | 3.343 | 0.4008 | 0.0009 |
| zsre_91 | Atreus | Darius III | 5.559 | 3.847 | 3.853 | 5.404 | 3.7987 | 0.0040 |
| zsre_93 | Bioscience Horizons | Wiley-Blackwell | 4.360 | 3.566 | 3.371 | 6.769 | 0.2924 | 0.0011 |
| zsre_94 | HD 125658 | Leo Minor | 7.171 | 4.206 | 3.329 | 6.825 | 1.2478 | 0.0007 |
| zsre_95 | Colorhythm | Lil' Mo | 5.287 | 3.717 | 4.582 | 6.691 | 2.1871 | 0.0012 |
| zsre_96 | Maria Antonia of Austria | Elisabeth of Bavaria | 2.880 | 2.446 | 2.556 | 5.878 | 0.4642 | 0.0005 |
| zsre_97 | Estate Exchange | Welton Becket | 5.557 | 3.212 | 3.481 | 3.708 | 1.4994 | 0.0020 |
| zsre_98 | Château Mont-Royal | Édouard Niermans | 4.122 | 1.622 | 1.017 | 3.675 | 0.8333 | 0.0005 |
| zsre_99 | Moses Magnum | Noon Universe | 5.971 | 1.168 | 1.040 | 4.668 | 0.4761 | 0.0009 |
| zsre_100 | Fritz X | 1940 | 3.880 | 1.969 | 1.048 | 3.344 | 1.0845 | 0.0019 |
| zsre_101 | Mongenast Ministry | 1941 | 4.133 | 0.967 | 1.005 | 2.143 | 1.0831 | 0.0014 |
| zsre_102 | Halenia | Geometridae | 4.860 | 0.485 | 0.605 | 4.592 | 1.2176 | 0.0023 |
| zsre_103 | Ariadne musica | orchestra | 5.469 | 5.651 | 5.487 | 2.806 | 2.5385 | 0.0007 |
| zsre_104 | Simple Souls | TSR | 8.778 | 5.787 | 5.800 | 4.781 | 0.2898 | 0.0012 |
| zsre_105 | Anhui musk deer | vulnerable | 6.342 | 4.483 | 3.688 | 4.520 | 0.4505 | 0.0008 |
| zsre_106 | Andrew Toney | Vancouver Canucks | 5.421 | 2.114 | 2.768 | 4.733 | 1.8078 | 0.0007 |
| zsre_107 | Horkos | Amenhotep III | 5.463 | 1.777 | 1.510 | 8.933 | 2.4397 | 0.0021 |
| zsre_108 | The Last Days | Peter Bogdanovich | 5.450 | 2.031 | 2.627 | 5.264 | 1.0289 | 0.0015 |
| zsre_109 | Fimpen | Wolfgang Becker | 5.582 | 1.345 | 1.881 | 3.831 | 1.0223 | 0.0011 |
| zsre_110 | JS 7.62 | 1966 | 3.162 | 1.642 | 1.762 | 3.795 | 1.5728 | 0.0010 |
| zsre_111 | Philipp Orter | 20 April 1894 | 3.031 | 1.851 | 1.815 | 4.563 | 0.0667 | 0.0017 |
| zsre_112 | Anguispira | Crambidae | 5.442 | 0.427 | 0.716 | 1.401 | 2.9517 | 0.0014 |
| zsre_113 | Melissa Magstadt | member of the Illinois House of Representatives | 3.088 | 2.081 | 2.064 | 4.099 | 0.1715 | 0.0016 |
| zsre_114 | MAT-49 | 2011 | 4.416 | 1.282 | 1.322 | 2.478 | 0.5134 | 0.0028 |
| zsre_115 | Fritz X | 1940 | 3.893 | 1.587 | 1.601 | 2.923 | 0.6905 | 0.0010 |
| zsre_116 | Josef Lada | Seville | 8.629 | 3.181 | 3.347 | 3.953 | 0.9072 | 0.0014 |
| zsre_117 | Fambly 42 | Warner Bros | 6.164 | 4.372 | 3.922 | 3.619 | 0.5646 | 0.0028 |
| zsre_118 | José Wilker | yellow fever | 8.313 | 4.706 | 4.559 | 2.619 | 0.2953 | 0.0040 |
| zsre_119 | Southern Crab Nebula | Cygnus | 4.544 | 2.957 | 3.207 | 5.899 | 2.6827 | 0.0014 |
| zsre_120 | Esther Bloom | The Divine Comedy | 6.547 | 5.310 | 3.558 | 3.129 | 1.5700 | 0.0018 |
| zsre_121 | British Rail Class 47 | Trimark | 10.306 | 1.967 | 2.870 | 9.478 | 0.3037 | 0.0023 |
| zsre_122 | Nikolaos Kriezotis | Greco-Italian War | 3.118 | 0.398 | 0.513 | 2.204 | 3.4336 | 0.0010 |
| zsre_123 | The Queen and the Dreams | Motown | 5.241 | 3.634 | 2.672 | 4.899 | 0.6825 | 0.0010 |
| zsre_124 | Antônio Cardoso | Rio Grande do Sul | 2.865 | 0.487 | 0.554 | 4.757 | 0.4677 | 0.0011 |
| zsre_125 | Lloyd Thomas | Spanish Civil War | 4.800 | 0.300 | 0.396 | 1.372 | 0.5062 | 0.0060 |
| zsre_126 | Angolan African dormouse | Lecithoceridae | 7.096 | 1.221 | 3.544 | 2.502 | 0.1103 | 0.0013 |
| zsre_127 | Lloyd Thomas | Spanish Civil War | 4.731 | 1.480 | 1.935 | 1.844 | 0.4923 | 0.0016 |
| zsre_128 | Leucyl/cystinyl aminopeptidase | male | 12.603 | 5.967 | 8.415 | 10.189 | 0.7827 | 0.0012 |
| zsre_129 | Cercestis | Noctuidae | 3.024 | 1.533 | 1.799 | 3.140 | 0.7138 | 0.0036 |
| zsre_130 | NGC 2 | Dorado | 7.224 | 6.628 | 7.420 | 7.057 | 3.0698 | 0.0008 |
| zsre_131 | Wolf 359 | Auriga | 4.289 | 1.970 | 4.026 | 3.843 | 4.5801 | 0.0006 |
| zsre_132 | Mohammad Naseem | Tajikistan | 3.349 | 0.682 | 1.880 | 2.468 | 1.3661 | 0.0006 |
| zsre_133 | United Abominations | Arista Records | 3.313 | 3.173 | 2.819 | 5.745 | 0.6475 | 0.0010 |
| zsre_134 | Beatriz Balzi | 17 May 2015 | 2.730 | 1.256 | 0.718 | 2.014 | 0.8643 | 0.0008 |
| zsre_135 | Prince Emmanuel of Belgium | Prince Philippe of Belgium | 2.719 | 0.462 | 1.393 | 3.524 | 0.5390 | 0.0007 |