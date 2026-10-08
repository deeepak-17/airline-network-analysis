"""Hand-built country -> continent mapping (no external package available).
Judgement calls: Russia split by longitude (lon>60 -> Asia, else Europe); Turkey, Cyprus:
Turkey->Asia, Cyprus->Europe; Caucasus & Central Asia->Asia; Egypt->Africa; Greenland, Caribbean,
Central America, Bermuda -> North America; Hawaii (inside 'United States') stays North America
(country-level mapping); Pacific island states/territories -> Oceania."""
_C = {
 "Europe": "Albania,Austria,Belarus,Belgium,Bosnia and Herzegovina,Bulgaria,Croatia,Cyprus,Czech Republic,Denmark,Estonia,Faroe Islands,Finland,France,Germany,Gibraltar,Greece,Guernsey,Hungary,Iceland,Ireland,Isle of Man,Italy,Jersey,Latvia,Lithuania,Luxembourg,Macedonia,Malta,Moldova,Montenegro,Netherlands,Norway,Poland,Portugal,Romania,Serbia,Slovakia,Slovenia,Spain,Sweden,Switzerland,Ukraine,United Kingdom",
 "Asia": "Afghanistan,Armenia,Azerbaijan,Bahrain,Bangladesh,Bhutan,Brunei,Burma,Cambodia,China,East Timor,Georgia,Hong Kong,India,Indonesia,Iran,Iraq,Israel,Japan,Jordan,Kazakhstan,Kuwait,Kyrgyzstan,Laos,Lebanon,Macau,Malaysia,Maldives,Mongolia,Nepal,North Korea,Oman,Pakistan,Philippines,Qatar,Saudi Arabia,Singapore,South Korea,Sri Lanka,Taiwan,Tajikistan,Thailand,Turkey,Turkmenistan,United Arab Emirates,Uzbekistan,Vietnam,Yemen,Christmas Island,Cocos (Keeling) Islands",
 "Africa": "Algeria,Angola,Benin,Botswana,Burkina Faso,Burundi,Cameroon,Cape Verde,Central African Republic,Chad,Comoros,Congo (Brazzaville),Congo (Kinshasa),Cote d'Ivoire,Djibouti,Egypt,Equatorial Guinea,Eritrea,Ethiopia,Gabon,Gambia,Ghana,Guinea,Guinea-Bissau,Kenya,Lesotho,Liberia,Libya,Madagascar,Malawi,Mali,Mauritania,Mauritius,Mayotte,Morocco,Mozambique,Namibia,Niger,Nigeria,Reunion,Rwanda,Sao Tome and Principe,Senegal,Seychelles,Sierra Leone,Somalia,South Africa,South Sudan,Sudan,Swaziland,Tanzania,Togo,Tunisia,Uganda,Western Sahara,Zambia,Zimbabwe",
 "North America": "Anguilla,Antigua and Barbuda,Aruba,Bahamas,Barbados,Belize,Bermuda,British Virgin Islands,Canada,Cayman Islands,Costa Rica,Cuba,Dominica,Dominican Republic,El Salvador,Greenland,Grenada,Guadeloupe,Guatemala,Haiti,Honduras,Jamaica,Martinique,Mexico,Netherlands Antilles,Nicaragua,Panama,Puerto Rico,Saint Kitts and Nevis,Saint Lucia,Saint Pierre and Miquelon,Saint Vincent and the Grenadines,Trinidad and Tobago,Turks and Caicos Islands,United States,Virgin Islands",
 "South America": "Argentina,Bolivia,Brazil,Chile,Colombia,Ecuador,Falkland Islands,French Guiana,Guyana,Paraguay,Peru,Suriname,Uruguay,Venezuela",
 "Oceania": "American Samoa,Australia,Cook Islands,Fiji,French Polynesia,Guam,Kiribati,Marshall Islands,Micronesia,Nauru,New Caledonia,New Zealand,Niue,Norfolk Island,Northern Mariana Islands,Palau,Papua New Guinea,Samoa,Solomon Islands,Tonga,Tuvalu,Vanuatu,Wallis and Futuna",
}
COUNTRY_TO_CONTINENT = {c: k for k, v in _C.items() for c in v.split(",")}

def continent_of(country, lon=None):
    if country == "Russia":
        return "Asia" if (lon is not None and lon > 60) else "Europe"
    return COUNTRY_TO_CONTINENT.get(country)
