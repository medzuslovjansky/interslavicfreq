import re

trans_tables = { 
    'isv': 'ć-č ć-č ć-č ś-s ź-z ŕ-r ĺ-l ľ-l ń-n t́-t ť-t d́-d ď-d đ-dž ò-o ȯ-o ė-e è-e č-č š-š ž-ž ě-ě е̌-ě å-a ę-e ų-u ě-e y-i ньј-nj ь- а-a ӑ-a б-b в-v ў-v г-g ґ-g д-d дж-dž ђ-dž е-e є-e ѣ-e ж-ž з-z и-i ј-j ї-ji й-j к-k л-l љ-lj м-m н-n њ-nj о-o п-p р-r с-s т-t у-u ф-f х-h ц-c ч-č ш-š щ-šč ъ-o ы-y ю-ju я-ja ё-e ѫ-u ѧ-e ћ-č ѥ-je ꙑ-i',
    'isv_to_slovianto': 'ě-e y-i',
    'isv_to_standard': 'ć-č ć-č ć-č ś-s ź-z ŕ-r ĺ-l ľ-l ń-n t́-t ť-t d́-d ď-d đ-dž ò-o ȯ-o ė-e è-e č-č š-š ž-ž ě-ě е̌-ě å-a ę-e ų-u',
    'isv_to_cyrillic': 'ń-н ľ-л nj-њ lj-љ ć-ч ć-ч ć-ч ś-с ź-з ŕ-р t́-т ť-т d́-д ď-д đ-дж ò-о ȯ-о ė-е è-е č-ч š-ш ž-ж ě-є е̌-є ě-є å-а ę-е ų-у a-а b-б c-ц č-ч d-д e-е f-ф g-г h-х i-и j-ј k-к l-л m-м n-н o-о p-п r-р s-с š-ш t-т u-у v-в y-ы z-з ž-ж',

    'ru': 'ё-е а́-а е́-е и́-и о́-о у́-у ы́-ы э́-э ю́-ю я́-я',
    'uk': 'ґ-г а́-а е́-е и́-и о́-о у́-у ы́-ы є́-є ю́-ю я́-я і́-і ї́-ї',  
    'be': 'ґ-г а́-а е́-е и́-и о́-о у́-у ы́-ы э́-э ю́-ю я́-я і́-і',  
    'bg': 'ѝ-и',
    'mk': 'ѝ-и ѐ-е',
    'kir_to_lat': 'ньј-ńj ь- а-a ӑ-å б-b в-v ў-v г-g ґ-g д-d дж-dž ђ-dž е-e є-ě ѣ-ě ж-ž з-z и-i ј-j ї-ji й-j к-k л-l љ-lj м-m н-n њ-nj о-o п-p р-r с-s т-t у-u ф-f х-h ц-c ч-č ш-š щ-šč ъ-ȯ ы-y ю-ju я-ja ё-e ѫ-ų ѧ-ę ћ-ć ѥ-je ꙑ-y',     
    'kirilicna_zamena': 'ру-ru бе-be ук-uk бг-bg мк-mk ср-sr ua-uk cz-cs ms-isv мс-isv обнови-obnovi',
}


def transliteration(text, lang):
    if lang not in trans_tables.keys():
        return text
    replaces = trans_tables[lang] 
    if not text.islower():
        replaces = replaces + " " + trans_tables[lang].upper()
    for i in replaces.split(' '):
        letters = i.split('-')
        text = text.replace(letters[0], letters[1])
    return text


def transliteration2(text, lang='kir_to_lat'):
    series = re.split(r"`", text)
    for i in range(0, len(series), 2):
        series[i] = transliteration(series[i], lang)
    return '`'.join(series)
