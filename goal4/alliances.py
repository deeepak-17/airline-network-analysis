"""HAND-BUILT alliance membership (airline IATA codes). Union of carriers that were members at
any point ~2010-2024, because the OpenFlights routes snapshot is undated/old; membership changed
over time (e.g. US merged into AA, AB bankrupt, AS/FJ/WY joined oneworld later). Not an official
dataset; carriers not listed -> 'None' (independent/LCC/regional/other alliance)."""
ALLIANCES = {
 "Star Alliance": "A3,AC,CA,AI,NZ,NH,OZ,OS,AV,SN,CM,OU,MS,ET,BR,LO,LH,SK,ZH,SQ,SA,LX,TP,TG,TK,UA,JP",
 "oneworld": "AA,BA,CX,AY,IB,JL,MH,QF,QR,RJ,S7,UL,LA,JJ,AS,FJ,WY,AB,US,KA,AT",
 "SkyTeam": "AF,KL,DL,AM,AR,AZ,CZ,KE,ME,MU,SU,KQ,RO,OK,SV,VN,UX,MF,GA,CI,AE",
}
AIRLINE_TO_ALLIANCE = {a: k for k, v in ALLIANCES.items() for a in v.split(",") if not a.endswith("_NO")}
def alliance_of(code):
    return AIRLINE_TO_ALLIANCE.get(code, "None")
