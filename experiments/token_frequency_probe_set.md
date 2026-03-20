# Token Frequency Copy Circuit — Probe Set

## Frequency Bands (by token ID, proxy for training frequency)

All probes use the same template with novel fictional entities:
```
<start_of_turn>user\nThe sacred material of [Entity] is [answer]. What is the sacred material of [Entity]?\n<end_of_turn>\n<start_of_turn>model\nThe sacred material of [Entity] is
```

### Band 1 — Very Rare (token ID > 130K)
| # | Answer | Token ID | Entity |
|---|--------|----------|--------|
| 1 | obsidian | 226956 | Zenthari |
| 2 | iridium | 226508 | Vorthane |
| 3 | onyx | 211505 | Kelmaris |
| 4 | fennel | 174942 | Dratheon |
| 5 | antimony | 173019 | Thessara |
| 6 | niobium | 153581 | Corvantis |

### Band 2 — Rare (token ID 50K–130K)
| # | Answer | Token ID | Entity |
|---|--------|----------|--------|
| 7 | elm | 94469 | Pyrantha |
| 8 | jade | 82947 | Solvenar |
| 9 | cedar | 80444 | Marethis |
| 10 | tungsten | 71090 | Jorvalis |
| 11 | amber | 67706 | Nexavorn |
| 12 | cobalt | 56896 | Zarkhona |

### Band 3 — Medium (token ID 15K–50K)
| # | Answer | Token ID | Entity |
|---|--------|----------|--------|
| 13 | pearl | 48361 | Kelthrim |
| 14 | platinum | 45449 | Brixara |
| 15 | mercury | 37291 | Velthane |
| 16 | coral | 35809 | Dravonis |
| 17 | oak | 32049 | Thessvarn |
| 18 | silk | 27373 | Mordaxis |

### Band 4 — Common (token ID 5K–15K)
| # | Answer | Token ID | Entity |
|---|--------|----------|--------|
| 19 | cotton | 14538 | Zelvoran |
| 20 | tin | 13657 | Crenthax |
| 21 | stone | 10810 | Korvathi |
| 22 | silver | 10173 | Pyrethis |
| 23 | iron | 8603 | Jexalorn |
| 24 | steel | 8103 | Thrennax |

### Band 5 — Very Common (token ID < 5K)
| # | Answer | Token ID | Entity |
|---|--------|----------|--------|
| 25 | gold | 5122 | Volthari |
| 26 | fire | 4304 | Nexthara |
| 27 | green | 3826 | Selvaris |
| 28 | blue | 3730 | Morthane |
| 29 | red | 2604 | Kelvaxis |
| 30 | water | 1813 | Draventh |

## Token ID Distribution
- Band 1 mean: 194,069
- Band 2 mean: 75,592
- Band 3 mean: 37,722
- Band 4 mean: 10,981
- Band 5 mean: 3,567
